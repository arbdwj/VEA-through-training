# VEA through training

Companion code, data, and figures for **"Tracing Eval-Awareness Emergence Through OLMo-3 Training"** — where in the training pipeline does verbalized evaluation-awareness (VEA) come from?

We measure VEA (Goodfire Appendix F1 rubric, applied to the chain-of-thought before `</think>`; a rollout is VEA if the judge returns a non-`None` quote) across the full OLMo-3 training pipeline — pretraining checkpoints → SFT → DPO → RLVR — on 5 safety benchmarks (444 prompts × 8 rollouts each). Headline findings:

## Reproduce the figures
Needs `numpy pandas matplotlib scipy` (and `pdflatex` for the PDF). Figures regenerate from the bundled `data/` alone — no GPU or API needed:
```bash
python -m venv .venv && ./.venv/bin/pip install numpy pandas matplotlib scipy
PY=./.venv/bin/python ./build.sh         # figures + PDF
# or a single figure:
cd scripts && ../.venv/bin/python fig1_pretrain_vea.py
```
