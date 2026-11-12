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
generate.py     one checkpoint      modal run generate.py --model rlvr --step 2300 [--sampling long_fortress]
sweep.py        range of ckpts      modal run --detach sweep.py --model rlvr [--steps 50,800]
judge.py        VEA / refusal       python judge.py vea --model rlvr
analyze.py      analyses + figures  python analyze.py template-gap | figures | ...
configs/
  models/         WHAT: pretrain, sft, dpo, rlvr. Identical fields in every file (schema-checked):
                  model, revision pattern, steps, output key, pinned template, dataset, default sampling
  tmp_configs/    ancillary experiments, same schema: sft_merge, rlvr_swap, rlvr_train (template gap)
  sampling/       HOW: n, temperature, top_p, seed, context, token budgets (one per sweep)
                  standard (8k)  long_fortress (8k, Fortress 16k)  pretrain (2k in 8k context)
  settings.yaml   judge model/effort, local output paths, Modal infra
vea/            library behind the four scripts
  config.py       YAML (with `extends:`) -> per-checkpoint job
  templates.py    pinned template, exact-string + one-token-swap assertions, token budgets
  modal_app.py    Modal functions: CPU preflight, GPU sampling (vLLM, TP=2)
  rubrics.py      Goodfire F1 VEA rubric, refusal rubric, CoT/answer extraction
  judging.py      resumable parallel judge runner (gpt-5-mini)
  analysis/       build_data, compare, template_gap, eval_gaming, pretrain_judges
  figures/        one script per figure (run via analyze.py figures)
  progress_server.py   live generation progress: python -m vea.progress_server
data/           corrected data (input to every figure), one folder per stage:
  prompts/        bench_prompts.jsonl, fortress_prompts.jsonl (the 444 prompts)
  pretrain/       rollouts/, vea.jsonl
  base/           vea.jsonl
  sft/            vea.jsonl (SFT steps + merge), refusal.jsonl, training_data_vea.jsonl
  dpo/            vea.jsonl, refusal.jsonl, training_data_vea.jsonl
  rlvr/           vea.jsonl, refusal.jsonl, vea_first_token.csv, vea_position*.csv
                  every row has stage, step, benchmark, prompt_id (Fortress ids are fortress_<n>)
figures/        corrected figures; v1_vs_v2/ = each published figure next to its fix
archive/v1/     as published: data/ (same stage layout), figures/, and code/ (v1-only one-offs, kept for provenance)
outputs/        raw rollouts and labels (gitignored; large)
```

A **model config** says which checkpoints to load (`revision: "step_{step:04d}"` over `steps:`), which
chat template replaces the branch's own (repo + pinned commit + the exact expected string), which
dataset, and which sampling config it uses by default. A **sampling config** is shared by the whole
sweep; `--sampling` overrides the default. Ancillary runs live in `configs/tmp_configs/` and are addressed as
`--model tmp_configs/rlvr_swap`; they differ from `models/rlvr.yaml` only in `stage`, `key`, `template`, `note`.

## Reproduce

Figures and analyses run from `data/` alone (no GPU or API):
```bash
pip install numpy pandas matplotlib scipy statsmodels pyyaml
python analyze.py figures                                     # all figures -> figures/
python analyze.py figures fig_template_gap                    # one figure
python analyze.py figures --data-dir archive/v1/data --fig-dir /tmp/v1   # as-published versions
python analyze.py template-gap
```
`fig8` needs refusal labels on all 5 benchmarks at a step; the corrected data only has Fortress refusal labels
for re-sampled steps, so it is skipped there and builds from `archive/v1/data`.

Full pipeline (Modal + `OPENAI_API_KEY`):
```bash
modal run sweep.py --model rlvr --preflight                 # CPU: render + assert every prompt
modal run --detach sweep.py --model rlvr                    # sample every step in the model config
python judge.py vea --model rlvr                            # pull from the volume + judge
python judge.py refusal --model rlvr
python analyze.py build-data && python analyze.py figures
```
