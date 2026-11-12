# Post figures: v1 (as published) vs v2 (template-fixed rerun)

`__v1` = exact image from the published LessWrong post. `__v2_updated` = regenerated from the corrected `data/`
(`analysis/build_data.py`), in which 10 SFT and 6 RLVR checkpoints were re-sampled with one pinned chat
template (`Olmo-3-32B-Think-DPO@9760402`) and every figure uses the gpt-5-mini judge.

| Post figure | What changed |
|---|---|
| Fig 1 pretraining | judge Gemini-2.5-flash -> gpt-5-mini; final ckpt 1.2% -> 0.7% of CoTs (conclusion unchanged) |
| Fig 2 RLVR by benchmark | steps 50/1050/1800/2300 re-sampled without the Ai2 system prompt; curve near-monotonic |
| Fig 3 stage-connected | SFT sweep ~halved; DPO no longer a drop; RLVR bumps gone |
| Fig 4 net change | DPO -17 -> +3 pts; SFT -10 -> -7; RLVR +23 -> +19 (DPO->2300) |
| Fig 5a matched triples | unchanged (training data, no template involved) |
| Fig 5b model vs data | model SFT->DPO 40%->22% becomes 19%->22% |
| Fig 6 eval-gaming | SFT/RLVR re-sampled ckpts relabelled; refusal rubric reconstructed (96.5% agreement with old labels on unchanged DPO) |

Regenerate: `./build.sh` (corrected) or `DATA_DIR=archive/v1/data FIG_DIR=<dir> ./build.sh` (as published).

`fig_template_gap_rlvr.png` is new in v2 (no v1 counterpart): VEA under the training template / one-token swap vs the eval template, SFT through RLVR.
