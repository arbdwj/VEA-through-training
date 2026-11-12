"""Modal app: GPU sampling for one checkpoint, and a CPU preflight that renders + asserts
every prompt. Infra settings come from configs/settings.yaml (`modal:`); everything about the
experiment comes in the job dict built by vea.config.jobs().
"""
import hashlib
import json
import os
import time
from pathlib import Path

import modal
import yaml

_SETTINGS = Path(__file__).resolve().parents[1] / "configs" / "settings.yaml"
M = yaml.safe_load(_SETTINGS.read_text())["modal"]

app = modal.App(M["app"])   # reuse the granted app name (new names lack volume write access)

_local = lambda img: (img.add_local_python_source("vea")
                      .add_local_file(str(_SETTINGS), "/root/configs/settings.yaml"))
gpu_image = _local(
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("vllm>=0.11.0", "huggingface_hub>=0.26", "transformers", "numpy", "tqdm")
    .env({"HF_HOME": "/cache", "HF_XET_HIGH_PERFORMANCE": "1",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "CUDA_HOME": "/usr/local/cuda",
          "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"}))
cpu_image = _local(modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub>=0.26", "transformers", "jinja2"))

weights_vol = modal.Volume.from_name(M["weights_volume"], create_if_missing=True)
outputs_vol = modal.Volume.from_name(M["outputs_volume"], create_if_missing=True)
HF_SECRET = modal.Secret.from_name(M["hf_secret"])
progress = modal.Dict.from_name(M["progress_dict"], create_if_missing=True)


def _hf_login():
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        if os.environ.get(var):
            os.environ["HF_TOKEN"] = os.environ[var]
            return


def _set(key, **fields):
    """Merge fields into this checkpoint's progress record (read by vea/progress_server.py)."""
    try:
        cur = progress.get(key) or {}
    except Exception:
        cur = {}
    cur.update(fields, updated=time.time())
    try:
        progress[key] = cur
    except Exception as e:
        print(f"[progress] write failed: {e}")


@app.function(image=cpu_image, secrets=[HF_SECRET], volumes={"/cache": weights_vol}, timeout=1800, cpu=2.0)
def preflight(jobs, prompts):
    """CPU only: load each checkpoint's tokenizer, force the pinned template, render + assert all prompts."""
    from vea import templates as T
    _hf_login()
    res = []
    for job in jobs:
        tok = T.load_tokenizer(job["model"], job["revision"], job["template"], cache_dir="/cache")
        rendered = T.render(tok, prompts, job["template"])
        bud = T.budgets(prompts, rendered, job["sampling"])
        res.append({"key": job["key"], "n": len(rendered), "max_prompt_tokens": max(len(i) for _, i in rendered),
                    "min_new_tokens": min(c for _, c in bud)})
    return {"results": res, "example": rendered[0][0]}


@app.function(image=gpu_image, secrets=[HF_SECRET], volumes={"/cache": weights_vol, "/outputs": outputs_vol},
              gpu=M["gpu"], timeout=int(M["timeout_hours"] * 3600))
def generate_ckpt(job, prompts, prompts_sha):
    from vea import templates as T
    key, s, spec = job["key"], job["sampling"], job["template"]
    t0 = time.time()
    _set(key, status="loading", started=t0, error=None)
    try:
        _hf_login()
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        from tqdm.auto import tqdm

        tok = T.load_tokenizer(job["model"], job["revision"], spec, cache_dir="/cache")
        rendered = T.render(tok, prompts, spec)            # fails fast, before loading weights
        bud = T.budgets(prompts, rendered, s)

        print(f"[{key}] loading {job['model']}@{job['revision']} (TP=2), template={spec['name']}")
        llm = LLM(model=job["model"], revision=job["revision"], tensor_parallel_size=2, dtype="bfloat16",
                  max_model_len=s["max_model_len"], gpu_memory_utilization=0.90, download_dir="/cache",
                  trust_remote_code=True, enforce_eager=False)
        weights_vol.commit()
        t_loaded = time.time()
        g, total = s["n"], len(prompts) * s["n"]
        _set(key, status="generating", loaded=t_loaded, load_s=round(t_loaded - t0), total=total, done=0)

        class _Progress(tqdm):
            """vLLM calls this as its progress bar; mirror it into the Dict every ~10s."""
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._last, self._incs = 0.0, []

            def update(self, n=1):
                r = super().update(n)
                self._incs.append(n)
                if time.time() - self._last > 10 or self.n >= (self.total or 0):
                    self._last = time.time()
                    # vLLM versions differ in whether total/update count requests or rollouts
                    done = self.n if (self.total == total or max(self._incs) >= g) else self.n * g
                    _set(key, done=min(int(done), total), postfix=str(self.postfix or ""))
                return r

        tps = [TokensPrompt(prompt_token_ids=ids) for ids, _ in bud]
        sps = [SamplingParams(n=g, temperature=s["temperature"], top_p=s["top_p"], max_tokens=cap,
                              seed=s["seed"], logprobs=None) for _, cap in bud]
        try:
            outs = llm.generate(tps, sps, use_tqdm=_Progress)
        except TypeError:
            outs = llm.generate(tps, sps)

        _set(key, status="writing", done=total)
        os.makedirs(job["out_dir"], exist_ok=True)
        path = f"{job['out_dir']}/{key}.jsonl"
        n = n_think = n_trunc = n_self_open = 0
        toks = []
        with open(path, "w") as f:
            for p, (text, _), (_, cap), o in zip(prompts, rendered, bud, outs):
                rsha = hashlib.sha256(text.encode()).hexdigest()
                for ri, c in enumerate(o.outputs):
                    n += 1
                    n_think += "</think>" in c.text
                    n_trunc += c.finish_reason == "length"
                    n_self_open += c.text.lstrip().startswith("<think>")
                    toks.append(len(c.token_ids))
                    row = {"stage": job["stage"], "lr": job["lr"], "step": job["step"],
                           "benchmark": p["benchmark"], "prompt_id": p["prompt_id"], "rollout_id": ri,
                           "prompt": p["prompt"], "completion": c.text, "n_tokens": len(c.token_ids),
                           "finish_reason": c.finish_reason, "model": job["model"], "revision": job["revision"],
                           "template": spec["name"], "template_repo": spec["repo"], "template_rev": spec["rev"],
                           "rendered_prompt": text, "rendered_prompt_sha256": rsha, "prompts_sha256": prompts_sha,
                           "temperature": s["temperature"], "top_p": s["top_p"], "seed": s["seed"],
                           "max_new_tokens": cap, "model_config": job["model_config"],
                           "sampling_config": job["sampling_config"]}
                    if "risk_domain" in p:
                        row["risk_domain"] = p["risk_domain"]
                    f.write(json.dumps(row) + "\n")
        outputs_vol.commit()
        toks.sort()
        stats = {"rollouts": n, "has_close_think": n_think, "length_capped": n_trunc,
                 "self_opened_think": n_self_open, "median_tokens": toks[len(toks) // 2], "path": path}
        _set(key, status="done", finished=time.time(), gen_s=round(time.time() - t_loaded), stats=stats)
        print(f"[{key}] done: {stats}")
        return stats
    except Exception as e:
        _set(key, status="failed", error=f"{type(e).__name__}: {str(e)[:500]}", finished=time.time())
        raise


def run_jobs(jobs, preflight_only=False):
    """Local side: preflight every job on CPU, then (unless preflight_only) sample them in parallel."""
    from vea import prompts as PR
    if len({json.dumps(j["dataset"], sort_keys=True) for j in jobs}) != 1:
        raise ValueError("all jobs in one run must share a dataset")
    prompts, sha = PR.load(jobs[0]["dataset"])
    print(f"model config: {jobs[0]['model_config']} | sampling config: {jobs[0]['sampling_config']}")
    print(f"{len(prompts)} prompts (sha {sha[:12]}), {len(jobs)} checkpoint(s): {', '.join(j['key'] for j in jobs)}")
    r = preflight.remote(jobs, prompts)
    for x in r["results"]:
        print(f"  preflight OK {x['key']}: {x['n']} prompts, max prompt {x['max_prompt_tokens']} tok, "
              f"min generation budget {x['min_new_tokens']} tok")
    print("example rendered prompt:\n" + r["example"])
    if preflight_only:
        return
    progress["_run"] = {"started": time.time(), "keys": [j["key"] for j in jobs], "n_prompts": len(prompts)}
    for j in jobs:
        progress[j["key"]] = {"status": "queued", "queued": time.time(), "updated": time.time(), "stage": j["stage"],
                              "step": j["step"], "model": j["model"], "revision": j["revision"],
                              "reason": j["note"], "total": len(prompts) * j["sampling"]["n"], "done": 0}
    handles = [(j["key"], generate_ckpt.spawn(j, prompts, sha)) for j in jobs]
    for k, h in handles:
        try:
            print(f"  {k}: {h.get()}")
        except Exception as e:
            print(f"  {k} FAILED: {type(e).__name__} {str(e)[:200]}")
