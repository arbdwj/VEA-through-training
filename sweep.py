"""Run generate over a range of checkpoints: every `steps:` entry of the model config, or a subset.
One sampling config applies to the whole sweep.

  modal run --detach sweep.py --model rlvr                       # all steps
  modal run --detach sweep.py --model rlvr --steps 50,800        # a subset
  modal run --detach sweep.py --model temp/rlvr_swap               # an ancillary experiment
  modal run sweep.py --model sft --preflight                     # CPU only: render + assert every checkpoint

Checkpoints run in parallel (as many as Modal grants); live progress: python -m vea.progress_server
"""
from vea import config as C
from vea.modal_app import app, run_jobs


@app.local_entrypoint()
def main(model: str, steps: str = "", sampling: str = "", preflight: bool = False):
    run_jobs(C.jobs(model, sampling or None, steps), preflight_only=preflight)
