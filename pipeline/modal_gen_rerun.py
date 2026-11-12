"""GENERATION stage of the template-fixed rerun (judging is a separate stage).

Why: the original sweep let each checkpoint use the chat template its HF revision
ships. SFT branch revisions (Olmo-3-32B-Think-SFT@5e-5-step*) ship a bare template
with no system turn and NO trailing <think>, so 97% of SFT rollouts opened <think>
themselves (shorter CoT, ~2x VEA). Four Olmo-3.1-32B-Think RLVR branches ship an
"You are Olmo ... built by Ai2" system prompt. Everything else (pretrain, DPO, the
other 6 RLVR steps) used the Olmo-3 Think eval template.

Fix: every checkpoint here renders prompts with ONE template, pinned by commit:
  allenai/Olmo-3-32B-Think-DPO @ 97604023f2602a7dfe6d31b1fc229985e0002022
and every rendered prompt is asserted to be exactly
  <|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n
  <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>
before any GPU time is spent generating. The rendered prompt is saved on every row.

Sampling matches the kept neighbours so reruns slot into the existing curves:
temperature 0.6, top_p 0.95, seed 42, 8 rollouts/prompt; max_new_tokens 8192
(SFT, and RLVR bench) / 16384 (RLVR fortress, as in the original RLVR run).

Outputs: olmo-outputs volume, /outputs/rerun_v2/<key>.jsonl (old files untouched).
Live progress: modal.Dict "vea-gen-progress" (see pipeline/progress_server.py).

Run:
  modal run pipeline/modal_gen_rerun.py --mode preflight          # CPU only, checks templates
  modal run --detach pipeline/modal_gen_rerun.py --mode sample    # all 14 checkpoints
  modal run --detach pipeline/modal_gen_rerun.py --mode sample --keys sft_5e-5_step10000
"""
import hashlib
import json
import os
import time

import modal

app = modal.App("olmo-vea-base")   # reuse the granted app name (new names lack volume write access)

TEMPLATE_REPO = "allenai/Olmo-3-32B-Think-DPO"
TEMPLATE_REV = "97604023f2602a7dfe6d31b1fc229985e0002022"
SYSTEM_DEFAULT = "You are a helpful AI assistant."
EXPECTED_SUFFIX = "<|im_start|>assistant\n<think>"
# Training ("SFT-branch") template, as shipped on Olmo-3-32B-Think-SFT@5e-5-step*: no system turn,
# no trailing <think> (the model writes <think> itself). Used only for template-gap jobs.
TRAIN_TEMPLATE_REPO = "allenai/Olmo-3-32B-Think-SFT"
TRAIN_TEMPLATE_REV = "5e-5-step10000"

G = 8
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 42
# Same context as the original DPO/RLVR runs (fortress_595 is 18.9k prompt tokens and got
# 20480-18929=1551 gen tokens there). max_new_tokens is clamped to the remaining context.
MAX_MODEL_LEN = 20480
OUT_DIR = "/outputs/rerun_v2"
PROGRESS_DICT = "vea-gen-progress"

SFT_REPO = "allenai/Olmo-3-32B-Think-SFT"
RLVR_REPO = "allenai/Olmo-3.1-32B-Think"


def _jobs():
    jobs = []
    for s in range(1000, 10001, 1000):
        jobs.append({"key": f"sft_5e-5_step{s:05d}", "stage": "sft", "lr": "5e-5", "step": s,
                     "model": SFT_REPO, "revision": f"5e-5-step{s}",
                     "cap_bench": 8192, "cap_fortress": 8192,
                     "reason": "SFT branch template: no system turn, no trailing <think>"})
    for s in (50, 1050, 1800, 2300):
        jobs.append({"key": f"rlvr_step_{s:04d}", "stage": "rlvr", "lr": "na", "step": s,
                     "model": RLVR_REPO, "revision": f"step_{s:04d}",
                     "cap_bench": 8192, "cap_fortress": 16384,
                     "reason": "RLVR branch template: 'You are Olmo ... Ai2' system prompt"})
    # Generic-template branches whose ORIGINAL rollout files had completions attached to the
    # wrong prompts (bench step 300: 91% mismatched; fortress step 800: 69%). Re-sampled whole.
    for s in (300, 800):
        jobs.append({"key": f"rlvr_step_{s:04d}", "stage": "rlvr", "lr": "na", "step": s,
                     "model": RLVR_REPO, "revision": f"step_{s:04d}",
                     "cap_bench": 8192, "cap_fortress": 16384,
                     "reason": "original rollout file misaligned (completions on wrong prompt ids)"})
    # Template-gap experiment: RLVR checkpoints under the TRAINING template (same as the
    # original SFT sweep), to test whether the ~20-pt template gap persists through RLVR.
    for s in (50, 800, 1550, 2300):
        jobs.append({"key": f"rlvrtrain_step_{s:04d}", "stage": "rlvr_traintmpl", "lr": "na", "step": s,
                     "model": RLVR_REPO, "revision": f"step_{s:04d}", "template": "train",
                     "cap_bench": 8192, "cap_fortress": 16384,
                     "reason": "template-gap: training template (no system, no <think>)"})
    # One-token swap (revisiting post's Fig 3 manipulation): eval template, final '>' -> '>\n'.
    for s in (50, 300, 550, 800, 1050, 1300, 1550, 1800, 2050, 2300):
        jobs.append({"key": f"rlvrswap_step_{s:04d}", "stage": "rlvr_swap", "lr": "na", "step": s,
                     "model": RLVR_REPO, "revision": f"step_{s:04d}", "template": "swap",
                     "cap_bench": 8192, "cap_fortress": 16384,
                     "reason": "template-gap: one-token swap '<think>' -> '<think>\\n'"})
    # Released SFT = 50/50 weight average of the 5e-5 and 1e-4 final checkpoints; DPO's actual
    # parent. Never sampled before. Caps match the DPO run (8192 for all benchmarks).
    jobs.append({"key": "sftmerge_main", "stage": "sft_merge", "lr": "merge", "step": 0,
                 "model": SFT_REPO, "revision": "a6d7f3cf497c7049712c13a664c65c7992f2da0c",
                 "cap_bench": 8192, "cap_fortress": 8192,
                 "reason": "new: released SFT (merge), DPO's starting point"})
    return jobs


JOBS = _jobs()
JOB_BY_KEY = {j["key"]: j for j in JOBS}

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("vllm>=0.11.0", "huggingface_hub>=0.26", "transformers", "numpy", "tqdm")
    .env({"HF_HOME": "/cache", "HF_XET_HIGH_PERFORMANCE": "1",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "CUDA_HOME": "/usr/local/cuda",
          "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
)
cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "huggingface_hub>=0.26", "transformers", "jinja2")
weights_vol = modal.Volume.from_name("olmo-weights", create_if_missing=True)
outputs_vol = modal.Volume.from_name("olmo-outputs", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface-secret")
progress = modal.Dict.from_name(PROGRESS_DICT, create_if_missing=True)


def _maybe_hf_login():
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            os.environ["HF_TOKEN"] = tok
            return


def _set(key, **fields):
    """Merge fields into this checkpoint's progress record."""
    try:
        cur = progress.get(key) or {}
    except Exception:
        cur = {}
    cur.update(fields, updated=time.time())
    try:
        progress[key] = cur
    except Exception as e:
        print(f"[progress] write failed: {e}")


def _load_tokenizer(job):
    """Checkpoint tokenizer with the pinned eval template forced on."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(job["model"], revision=job["revision"], cache_dir="/cache")
    if job.get("template") == "train":
        tmpl = AutoTokenizer.from_pretrained(TRAIN_TEMPLATE_REPO, revision=TRAIN_TEMPLATE_REV, cache_dir="/cache")
    else:
        tmpl = AutoTokenizer.from_pretrained(TEMPLATE_REPO, revision=TEMPLATE_REV, cache_dir="/cache")
    tok.chat_template = tmpl.chat_template
    return tok


def _render_all(tok, prompts, template="eval"):
    """Render + assert every prompt. Returns list of (text, ids)."""
    out = []
    for p in prompts:
        msgs = [{"role": "user", "content": p["prompt"]}]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if template == "train":
            expected = f"<|im_start|>user\n{p['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        else:
            expected = (f"<|im_start|>system\n{SYSTEM_DEFAULT}<|im_end|>\n"
                        f"<|im_start|>user\n{p['prompt']}<|im_end|>\n{EXPECTED_SUFFIX}")
        if template == "swap":
            # one-token swap: eval template, final '>' token replaced by the merged '>\n' token
            base_ids = tok(text, add_special_tokens=False)["input_ids"]
            text, expected = text + "\n", expected + "\n"
        if text != expected:
            raise AssertionError(f"rendered prompt mismatch for {p['prompt_id']}:\n{text[:300]!r}\n"
                                 f"--- expected ---\n{expected[:300]!r}")
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if template == "swap":
            nl = tok(">\n", add_special_tokens=False)["input_ids"]
            gt = tok(">", add_special_tokens=False)["input_ids"]
            if not (len(nl) == 1 and len(gt) == 1 and base_ids[-1] == gt[0] and ids[-1] == nl[0]
                    and ids[:-1] == base_ids[:-1]):
                raise AssertionError(f"one-token swap failed for {p['prompt_id']}: "
                                     f"{tok.convert_ids_to_tokens(base_ids[-2:])} -> {tok.convert_ids_to_tokens(ids[-2:])}")
        if tok.decode(ids) != text:
            raise AssertionError(f"token round-trip mismatch for {p['prompt_id']}")
        out.append((text, ids))
    return out


@app.function(image=cpu_image, secrets=[HF_SECRET], volumes={"/cache": weights_vol},
              timeout=1800, cpu=2.0)
def preflight(job_keys, prompts):
    """CPU-only: load each checkpoint's tokenizer, force the template, render+assert all prompts."""
    _maybe_hf_login()
    res = []
    for k in job_keys:
        job = JOB_BY_KEY[k]
        tok = _load_tokenizer(job)
        rendered = _render_all(tok, prompts, job.get("template", "eval"))
        lens = [len(ids) for _, ids in rendered]
        res.append({"key": k, "ok": True, "n": len(rendered), "max_prompt_tokens": max(lens)})
    return {"results": res, "example": rendered[0][0]}


@app.function(image=image, secrets=[HF_SECRET],
              volumes={"/cache": weights_vol, "/outputs": outputs_vol},
              gpu="H100:2", timeout=5 * 3600)
def generate_ckpt(key, prompts, prompts_sha, g=G):
    job = JOB_BY_KEY[key]
    t0 = time.time()
    _set(key, status="loading", started=t0, error=None)
    try:
        _maybe_hf_login()
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
        from tqdm.auto import tqdm

        tok = _load_tokenizer(job)
        rendered = _render_all(tok, prompts, job.get("template", "eval"))          # fails fast, before loading weights
        caps = [min(job["cap_fortress"] if p["benchmark"] == "fortress" else job["cap_bench"],
                    MAX_MODEL_LEN - len(ids)) for p, (_, ids) in zip(prompts, rendered)]
        if min(caps) < 256:
            raise AssertionError(f"a prompt leaves <256 tokens of context (min cap {min(caps)})")

        print(f"[{key}] loading {job['model']}@{job['revision']} (TP=2)")
        llm = LLM(model=job["model"], revision=job["revision"], tensor_parallel_size=2,
                  dtype="bfloat16", max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=0.90,
                  download_dir="/cache", trust_remote_code=True, enforce_eager=False)
        weights_vol.commit()
        t_loaded = time.time()
        total_rollouts = len(prompts) * g
        _set(key, status="generating", loaded=t_loaded, load_s=round(t_loaded - t0),
             total=total_rollouts, done=0)

        n_prompts = len(prompts)

        class _Progress(tqdm):
            """vLLM calls this as its progress bar; mirror it into the Dict every ~10s."""
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._last = 0.0
                self._incs = []

            def update(self, n=1):
                r = super().update(n)
                self._incs.append(n)
                now = time.time()
                if now - self._last > 10 or self.n >= (self.total or 0):
                    self._last = now
                    # vLLM versions differ: total may count parent requests or rollouts,
                    # and update() may add 1 or n per finished request. Normalise to rollouts.
                    if self.total == total_rollouts or max(self._incs) >= g:
                        done = self.n
                    else:
                        done = self.n * g
                    _set(key, done=min(int(done), total_rollouts), postfix=str(self.postfix or ""))
                return r

        tps = [TokensPrompt(prompt_token_ids=ids) for _, ids in rendered]
        sps = [SamplingParams(n=g, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=cap,
                              seed=SEED, logprobs=None) for cap in caps]
        try:
            outs = llm.generate(tps, sps, use_tqdm=_Progress)
        except TypeError:
            print("[warn] vLLM rejected callable use_tqdm; progress will jump at the end")
            outs = llm.generate(tps, sps)

        _set(key, status="writing", done=total_rollouts)
        os.makedirs(OUT_DIR, exist_ok=True)
        path = f"{OUT_DIR}/{key}.jsonl"
        n = n_think = n_trunc = n_self_open = 0
        toks = []
        with open(path, "w") as f:
            for p, (text, _), cap, o in zip(prompts, rendered, caps, outs):
                rsha = hashlib.sha256(text.encode()).hexdigest()
                for ri, c in enumerate(o.outputs):
                    comp = c.text
                    n += 1
                    n_think += "</think>" in comp
                    n_trunc += c.finish_reason == "length"
                    n_self_open += comp.lstrip().startswith("<think>")
                    toks.append(len(c.token_ids))
                    row = {"stage": job["stage"], "lr": job["lr"], "step": job["step"],
                           "benchmark": p["benchmark"], "prompt_id": p["prompt_id"], "rollout_id": ri,
                           "prompt": p["prompt"], "completion": comp, "n_tokens": len(c.token_ids),
                           "finish_reason": c.finish_reason,
                           "model": job["model"], "revision": job["revision"],
                           "template": job.get("template", "eval"),
                           "template_repo": TRAIN_TEMPLATE_REPO if job.get("template") == "train" else TEMPLATE_REPO,
                           "template_rev": TRAIN_TEMPLATE_REV if job.get("template") == "train" else TEMPLATE_REV,
                           "rendered_prompt": text, "rendered_prompt_sha256": rsha,
                           "prompts_sha256": prompts_sha, "temperature": TEMPERATURE,
                           "top_p": TOP_P, "seed": SEED, "max_new_tokens": cap}
                    for extra in ("risk_domain",):
                        if extra in p:
                            row[extra] = p[extra]
                    f.write(json.dumps(row) + "\n")
        outputs_vol.commit()
        toks.sort()
        stats = {"rollouts": n, "has_close_think": n_think, "length_capped": n_trunc,
                 "self_opened_think": n_self_open, "median_tokens": toks[len(toks) // 2],
                 "path": path}
        _set(key, status="done", finished=time.time(), gen_s=round(time.time() - t_loaded), stats=stats)
        print(f"[{key}] done: {stats}")
        return stats
    except Exception as e:
        _set(key, status="failed", error=f"{type(e).__name__}: {str(e)[:500]}", finished=time.time())
        raise


def _read_prompts(repo_root):
    prompts = []
    with open(os.path.join(repo_root, "data/prompts/bench_prompts.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            prompts.append({"prompt_id": r["prompt_id"], "benchmark": r["benchmark"], "prompt": r["prompt"]})
    with open(os.path.join(repo_root, "data/prompts/fortress_prompts.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            p = {"prompt_id": f"fortress_{r['prompt_id']}", "benchmark": "fortress", "prompt": r["prompt"]}
            if "risk_domain" in r:
                p["risk_domain"] = r["risk_domain"]
            prompts.append(p)
    sha = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    return prompts, sha


@app.local_entrypoint()
def main(mode: str = "preflight", keys: str = ""):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts, sha = _read_prompts(repo_root)
    chosen = [k.strip() for k in keys.split(",") if k.strip()] or [j["key"] for j in JOBS]
    for k in chosen:
        if k not in JOB_BY_KEY:
            raise ValueError(f"unknown key {k}; valid: {list(JOB_BY_KEY)}")
    print(f"{len(prompts)} prompts (sha {sha[:12]}), {len(chosen)} checkpoints")

    if mode == "preflight":
        r = preflight.remote(chosen, prompts)
        for x in r["results"]:
            print(f"  OK {x['key']}: {x['n']} prompts rendered, max prompt {x['max_prompt_tokens']} tok")
        print("example rendered prompt:\n" + r["example"])
        return

    if mode == "sample":
        progress["_run"] = {"started": time.time(), "keys": chosen, "g": G,
                            "template": f"{TEMPLATE_REPO}@{TEMPLATE_REV[:12]}",
                            "n_prompts": len(prompts)}
        for k in chosen:
            j = JOB_BY_KEY[k]
            progress[k] = {"status": "queued", "queued": time.time(), "updated": time.time(),
                           "stage": j["stage"], "step": j["step"], "model": j["model"],
                           "revision": j["revision"], "reason": j["reason"],
                           "total": len(prompts) * G, "done": 0}
        handles = [(k, generate_ckpt.spawn(k, prompts, sha)) for k in chosen]
        print("spawned:", ", ".join(k for k, _ in handles))
        for k, h in handles:
            try:
                print(f"  {k}: {h.get()}")
            except Exception as e:
                print(f"  {k} FAILED: {type(e).__name__} {str(e)[:200]}")
        return
    raise ValueError(f"unknown mode: {mode}")
