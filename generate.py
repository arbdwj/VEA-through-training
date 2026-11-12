"""Generate rollouts for ONE checkpoint.

  modal run generate.py --model rlvr_swap --step 2300
  modal run generate.py --model sft --step 10000 --sampling long_fortress   # override the model's default sampling
  modal run generate.py --model sft --step 10000 --preflight               # CPU only: render + assert

--model    configs/models/<name>.yaml    which checkpoint, template, dataset
--sampling configs/sampling/<name>.yaml  how to sample (default: the one named in the model config)

The prompt is rendered with the template pinned in the model config (never the branch's own),
asserted character-for-character, and saved on every output row. Output:
<settings.modal.out_dir>/<key>.jsonl on the outputs volume (pull + judge with judge.py).
"""
from vea import config as C
from vea.modal_app import app, run_jobs


@app.local_entrypoint()
def main(model: str, step: int, sampling: str = "", preflight: bool = False):
    run_jobs(C.jobs(model, sampling or None, [step]), preflight_only=preflight)
