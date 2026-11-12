"""Generate rollouts for ONE checkpoint described by a YAML config.

  modal run generate.py --config configs/rlvr_swap.yaml --step 2300
  modal run generate.py --config configs/sft.yaml --step 10000 --preflight   # CPU only: render + assert

The prompt is rendered with the template pinned in the config (never the branch's own),
asserted character-for-character, and saved on every output row. Output:
<modal.out_dir>/<key>.jsonl on the outputs volume (pull + judge with judge.py).
"""
from vea import config as C, prompts as PR
from vea.modal_app import app, run_jobs


@app.local_entrypoint()
def main(config: str, step: int, preflight: bool = False):
    cfg = C.load(config)
    prompts, sha = PR.load(cfg)
    run_jobs([C.job(cfg, step)], prompts, sha, preflight_only=preflight)
