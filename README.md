# VEA through training

Companion code, data, and figures for **"Tracing Eval-Awareness Emergence Through OLMo-3 Training"** — where in the training pipeline does verbalized evaluation-awareness (VEA) come from?

We measure VEA (Goodfire Appendix F1 rubric, applied to the chain-of-thought before `</think>`; a rollout is VEA if the judge returns a non-`None` quote) across the full OLMo-3 training pipeline — pretraining checkpoints → SFT → DPO → RLVR — on 5 safety benchmarks (444 prompts × 8 rollouts each). Headline findings:

- **VEA is essentially absent during pretraining (~1%)** — it is installed by post-training.
- Across post-training it is **installed by SFT, stripped by DPO, and rebuilt/amplified by RLVR**.
- Behavioral eval-gaming (refusing more when eval-aware), measured **within-question**, is small and not individually significant; pooled estimates overstate it (matches Goodfire's within-question +3–18 pp with our +4–20 pp).
- The first VEA token lands in the first third of the CoT (median ~187 tokens) but is **broad / stochastic** — no sharp positional trigger; the only clean structure is that **blunter prompts trip VEA earlier**.

The write-up is in [`post.md`](post.md) (LessWrong-formatted Markdown) and [`paper/vea_training.pdf`](paper/vea_training.pdf) (formal version).

## Layout
```
post.md                  # the write-up (Markdown, for LessWrong)
paper/                   # formal version: vea_training.tex + .pdf
figures/                 # fig1–fig8
figures_pct/             # same figures, also annotating % of the final training step on the x-axis
scripts/                 # one self-contained script per figure (read ../data, write ../figures)
  _paths.py  _stats.py   # path resolution + by-prompt cluster bootstrap
  _axes.py               # "% of final step" secondary axis / tick labels
  fig1_pretrain_vea.py ... fig8_vea_eval_gaming_split.py
data/                    # judgments, derived CSVs, prompts, pretraining rollouts
pipeline/                # upstream sampling/judging/analysis (Modal + OpenRouter; reference)
build.sh                 # regenerate all figures + compile the PDF
```

Each step-based figure shows the absolute training step **and** the percentage of that run's
final step (pct = 100·step / final; RLVR /2300, SFT /10000, pretraining /656000). The
`figures_pct/` copies are produced by re-running the scripts with `FIG_DIR=figures_pct`.

## Reproduce the figures
Needs `numpy pandas matplotlib scipy` (and `pdflatex` for the PDF). Figures regenerate from the bundled `data/` alone — no GPU or API needed:
```bash
python -m venv .venv && ./.venv/bin/pip install numpy pandas matplotlib scipy
PY=./.venv/bin/python ./build.sh         # figures + PDF
# or a single figure:
cd scripts && ../.venv/bin/python fig1_pretrain_vea.py
```
