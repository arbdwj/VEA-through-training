# VEA through training

Companion code, data, and figures for **"Tracing Eval-Awareness Emergence Through OLMo-3 Training"**: where in the OLMo-3 training pipeline does verbalized evaluation-awareness (VEA) come from?

VEA is judged with the Goodfire Appendix F1 rubric on the chain-of-thought before `</think>` (gpt-5-mini, reasoning effort low; a rollout is VEA if the judge returns a quote). We sample pretraining checkpoints → SFT → DPO → RLVR on 5 safety benchmarks (444 prompts × 8 rollouts).

## Correction (v2)

The published version sampled checkpoints with **whatever chat template each Hugging Face branch shipped**, so the prompt changed along with the weights:

| Stage | Template in v1 | Prompt ended with |
|---|---|---|
| Pretraining, DPO | generic eval template | `system: You are a helpful AI assistant.` … `<think>` |
| SFT (all steps) | SFT-branch training template | no system turn … `assistant\n` (model writes `<think>` itself) |
| RLVR, 24 of 46 branches | Ai2 identity system prompt | `system: You are Olmo … built by Ai2 …` … `<think>` |

v2 re-samples every affected checkpoint with one pinned template (`allenai/Olmo-3-32B-Think-DPO@9760402`), asserts the rendered prompt string, and records it in every row. The SFT template roughly doubled VEA (~2.1×), which created the v1 "DPO collapses VEA" result; the RLVR system-prompt switching created the synchronised benchmark wiggles. See [`figures/v1_vs_v2/`](figures/v1_vs_v2/) for every published figure next to its corrected version.

## Layout

```
data/             corrected data (default input for every figure)
archive/v1/       as published: data/ and figures/
figures/          corrected figures; v1_vs_v2/ = side-by-side with the published post
plotting/         one script per figure (+ _paths, _stats, _axes helpers)
generation/       Modal GPU sampling
  gen_checkpoints.py   SFT / merge / DPO / RLVR checkpoints, pinned + asserted template
                       (template = eval | train | swap for the template-gap experiment)
  gen_pretrain.py      pretraining checkpoints of the base model
  progress_server.py   local progress UI for gen_checkpoints (reads a modal.Dict)
judging/
  judge_vea.py         rubric, CoT extraction (shared library)
  judge_vea_rerun.py   VEA judge for gen_checkpoints outputs (gpt-5-mini)
  judge_refusal.py     refusal judge (decline-only rubric v2)
  judge_pretrain.py    gpt-5-mini re-judge of pretraining rollouts
  judge_pretrain_gemini_v1.py, refine_vea_first_token.py
analysis/
  build_data.py        archive/v1/data + rerun judgments -> data/
  template_gap.py      training-vs-eval template gap through SFT and RLVR
  eval_gaming_trend.py conditional-logit test of refusal ~ VEA × step
  vea_position*.py, compare_sft_dpo_data.py
outputs/          raw rollouts and judgments (gitignored; large)
```

## Reproduce

Figures regenerate from `data/` alone, with no GPU or API needed:
```bash
pip install numpy pandas matplotlib scipy
PY=python ./build.sh                                          # all figures -> figures/
DATA_DIR=archive/v1/data FIG_DIR=/tmp/v1 PY=python ./build.sh # as-published versions
cd plotting && python fig_template_gap.py                     # one figure
```
`fig8` needs refusal labels on all 5 benchmarks at a step; the corrected data only has Fortress refusal labels for re-sampled steps, so it is skipped there and builds from `archive/v1/data`.

Re-running the pipeline (needs Modal + `OPENAI_API_KEY`):
```bash
modal run generation/gen_checkpoints.py --mode preflight --keys sft_5e-5_step10000   # CPU: render + assert templates
modal run --detach generation/gen_checkpoints.py --mode sample --keys sft_5e-5_step10000
python judging/judge_vea_rerun.py judge --keys sft_5e-5_step10000
python analysis/build_data.py
```
