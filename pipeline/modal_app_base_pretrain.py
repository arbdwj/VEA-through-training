"""VEA across PRETRAINING checkpoints of the base model allenai/Olmo-3-1125-32B.

10 evenly-spaced stage1 checkpoints (steps 0..656000). The base has no chat template,
so we borrow the Think-SFT template (same as the step-0/base run) and sample the SAME
444 prompts. Early pretraining checkpoints may produce degenerate / no-<think> output —
the </think> rate per checkpoint is logged as a coherence diagnostic.

Output schema matches the stage files (stage="pretrain", step=<pretrain step>) so the
existing judges/plotters can consume it.

Run:
  modal run src/modal_app_base_pretrain.py --mode smoke           # 1 mid checkpoint
  modal run --detach src/modal_app_base_pretrain.py --mode sample # all 10
"""
import json
import os

import modal

app = modal.App("olmo-vea-base")   # reuse the granted app name (new names lack volume write access)

MODEL_BASE = "allenai/Olmo-3-1125-32B"
TEMPLATE_SRC = "allenai/Olmo-3-32B-Think-SFT"
CHECKPOINTS = [0, 80000, 155000, 228000, 301000, 371000, 444000, 517000, 585000, 656000]
G = 8
# Base pretraining checkpoints have no clean stop behavior — they ramble to the cap on
# nearly every rollout. VEA is verbalized early in the CoT (before </think>), so a smaller
# cap loses almost nothing for VEA detection while cutting cost/time ~4x vs 8192.
MAX_NEW_TOKENS = 2048
# Intermediate stage1-step* checkpoints have native max_position_embeddings=8192
# (the post-trained Think models and the base `main` revision are context-extended to
# 20480, but these pretraining checkpoints predate that). 8192 is their true context.
MAX_MODEL_LEN = 8192
MIN_GEN_BUDGET = 256
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 42

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .pip_install("vllm>=0.11.0", "huggingface_hub>=0.26", "transformers", "numpy", "tqdm")
    .env({"HF_HOME": "/cache", "HF_XET_HIGH_PERFORMANCE": "1",
          "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "CUDA_HOME": "/usr/local/cuda",
          "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"})
)
weights_vol = modal.Volume.from_name("olmo-weights", create_if_missing=True)
outputs_vol = modal.Volume.from_name("olmo-outputs", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface-secret")


def _maybe_hf_login():
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN",
                "HUGGINGFACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        tok = os.environ.get(var)
        if tok:
            os.environ["HF_TOKEN"] = tok
            try:
                from huggingface_hub import login
                login(token=tok, add_to_git_credential=False)
            except Exception as e:
                print(f"[hf] login error: {e}")
            return


@app.cls(image=image, secrets=[HF_SECRET],
         volumes={"/cache": weights_vol, "/outputs": outputs_vol},
         gpu="H100:2", timeout=5 * 3600, scaledown_window=120)
class BasePretrainRunner:
    revision: str = modal.parameter()
    max_model_len: int = modal.parameter(default=MAX_MODEL_LEN)

    @modal.enter()
    def load(self):
        from vllm import LLM
        from transformers import AutoTokenizer
        _maybe_hf_login()
        try:
            from vllm.v1.engine import input_processor as _ip
            _ip.InputProcessor._validate_model_input = lambda *a, **kw: None
            _ip.InputProcessor._validate_model_inputs = lambda *a, **kw: None
        except Exception as e:
            print(f"[warn] patch validator: {e}")
        self.tok = AutoTokenizer.from_pretrained(MODEL_BASE, revision=self.revision, cache_dir="/cache")
        tmpl = AutoTokenizer.from_pretrained(TEMPLATE_SRC, cache_dir="/cache")
        self.tok.chat_template = tmpl.chat_template
        print(f"[BasePretrainRunner] loading {MODEL_BASE}@{self.revision} (TP=2)")
        self.llm = LLM(model=MODEL_BASE, revision=self.revision, tensor_parallel_size=2,
                       dtype="bfloat16", max_model_len=self.max_model_len,
                       gpu_memory_utilization=0.90, download_dir="/cache",
                       trust_remote_code=True, enforce_eager=False)
        weights_vol.commit()
        print(f"[BasePretrainRunner] loaded {self.revision}")

    def _render(self, prompt):
        result = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                              add_generation_prompt=True, tokenize=True, return_dict=False)
        if isinstance(result, dict):
            result = result["input_ids"]
        try:
            import torch
            if isinstance(result, torch.Tensor):
                result = result.squeeze().tolist()
        except Exception:
            pass
        if result and isinstance(result[0], list):
            result = result[0]
        ids = [int(x) for x in result]
        cap = self.max_model_len - MIN_GEN_BUDGET
        return ids[:cap] if len(ids) > cap else ids

    @modal.method()
    def generate(self, prompts, n, temperature, top_p, max_new_tokens, seed=SEED):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        tp = [TokensPrompt(prompt_token_ids=self._render(p)) for p in prompts]
        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                            max_tokens=max_new_tokens, seed=seed, logprobs=None)
        outs = self.llm.generate(tp, sp)
        return [[{"text": c.text, "n_tokens": len(c.token_ids),
                  "finish_reason": c.finish_reason} for c in o.outputs] for o in outs]


def _load_prompts():
    prompts = []
    if os.path.exists("/outputs/bench_prompts.jsonl"):
        for l in open("/outputs/bench_prompts.jsonl"):
            r = json.loads(l)
            prompts.append({"prompt_id": r["prompt_id"], "benchmark": r["benchmark"], "prompt": r["prompt"]})
    if os.path.exists("/outputs/fortress_prompts.jsonl"):
        for l in open("/outputs/fortress_prompts.jsonl"):
            r = json.loads(l)
            prompts.append({"prompt_id": f"fortress_{r['prompt_id']}", "benchmark": "fortress", "prompt": r["prompt"]})
    return prompts


@app.function(image=image, secrets=[HF_SECRET],
              volumes={"/cache": weights_vol, "/outputs": outputs_vol},
              timeout=6 * 3600, cpu=2.0)
def sample_ckpt(step: int, g: int = G, max_new_tokens: int = MAX_NEW_TOKENS):
    rev = f"stage1-step{step}"
    print(f"[sample_ckpt] {rev}")
    runner = BasePretrainRunner(revision=rev, max_model_len=MAX_MODEL_LEN)
    prompts = _load_prompts()
    if not prompts:
        raise RuntimeError("no prompts on volume")
    texts = [p["prompt"] for p in prompts]
    gens = runner.generate.remote(texts, g, TEMPERATURE, TOP_P, max_new_tokens, SEED)
    out_path = f"/outputs/stage_rollouts_pretrain_step{step:07d}.jsonl"
    n, n_think, n_trunc = 0, 0, 0
    with open(out_path, "w") as f:
        for prow, group in zip(prompts, gens):
            for ri, comp in enumerate(group):
                if comp["finish_reason"] == "length":
                    n_trunc += 1
                if "</think>" in comp["text"]:
                    n_think += 1
                f.write(json.dumps({
                    "stage": "pretrain", "lr": "na", "step": step, "benchmark": prow["benchmark"],
                    "prompt_id": prow["prompt_id"], "rollout_id": ri, "prompt": prow["prompt"],
                    "completion": comp["text"], "n_tokens": comp["n_tokens"],
                    "finish_reason": comp["finish_reason"]}) + "\n")
                n += 1
    outputs_vol.commit()
    print(f"[sample_ckpt] {rev}: wrote {n} ({n_think} with </think>, {n_trunc} length-capped) -> {out_path}")
    return step, n_think, n


@app.local_entrypoint()
def main(mode: str = "smoke", steps: str = ""):
    chosen = [int(s) for s in steps.split(",") if s.strip()] if steps else CHECKPOINTS
    if mode == "smoke":
        s = 301000   # mid-pretraining: most likely to produce usable CoT
        print(f"=== smoke: pretrain stage1-step{s} ===")
        step, nt, n = sample_ckpt.remote(s)
        print(f"done: step {step}  {nt}/{n} with </think>")
        return
    if mode == "sample":
        print(f"=== sampling {len(chosen)} pretraining checkpoints in parallel ===")
        handles = [sample_ckpt.spawn(s) for s in chosen]
        for s, h in zip(chosen, handles):
            try:
                step, nt, n = h.get()
                print(f"  step {step}: {nt}/{n} with </think>")
            except Exception as e:
                print(f"  step {s} FAILED: {type(e).__name__} {str(e)[:140]}")
        print("Next: judge (VEA) + plot the pretraining trend.")
        return
    raise ValueError(f"unknown mode: {mode}")
