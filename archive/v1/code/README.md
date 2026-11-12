# v1 code (provenance only)

Scripts from the published (v1) analysis whose outputs are frozen in `data/` and that the
current pipeline does not re-run. They are kept as written and expect the v1 `outputs/` and flat `data/` layout
(and an OpenRouter key), so they will not run unmodified from here.

| Script | Produced |
|---|---|
| `gen_pretrain.py` | pretraining rollouts (`data/pretrain/rollouts/`); superseded by `configs/models/pretrain.yaml` + `configs/sampling/pretrain.yaml` |
| `judge_vea.py` | v1 VEA judge (OpenRouter); its rubric now lives in `vea/rubrics.py` |
| `judge_pretrain_gemini_v1.py` | v1 pretraining labels (Gemini-2.5-flash), `archive/v1/data/pretrain/vea.jsonl` |
| `refine_vea_first_token.py` | `data/rlvr/vea_first_token.csv` (Fig 6, 7) |
| `vea_position.py`, `vea_position_confound.py` | `data/rlvr/vea_position*.csv` |
| `compare_sft_dpo_data.py` | `data/{sft,dpo}/training_data_vea.jsonl` (Fig 9, Fig 5b data bars) |
