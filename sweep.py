"""Run generate over a range of checkpoints (all `steps:` in the config, or a subset).

  modal run --detach sweep.py --config configs/rlvr_swap.yaml                 # every step in the config
  modal run --detach sweep.py --config configs/rlvr_swap.yaml --steps 50,800  # a subset
  modal run sweep.py --config configs/sft.yaml --preflight                    # CPU only: render + assert all

Checkpoints run in parallel (as many as Modal grants); live progress: python -m vea.progress_server
"""
from vea import config as C, prompts as PR
from vea.modal_app import app, run_jobs


@app.local_entrypoint()
def main(config: str, steps: str = "", preflight: bool = False):
    cfg = C.load(config)
    prompts, sha = PR.load(cfg)
    run_jobs(C.jobs(cfg, steps), prompts, sha, preflight_only=preflight)
