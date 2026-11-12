#!/usr/bin/env bash
# Regenerate every figure in figures/ from the bundled data/ (no GPU or API needed).
#   PY=/path/to/python ./build.sh                                    # corrected data (default)
#   DATA_DIR=archive/v1/data FIG_DIR=/tmp/v1 PY=... ./build.sh       # as-published data
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"   # needs numpy pandas matplotlib scipy

cd "$HERE/plotting"
export FIG1_YMAX="${FIG1_YMAX:-3.2}"   # same y-range as the published Fig 1
for s in fig1_pretrain_vea fig2_rlvr_vea_benchmarks fig3_stage_connected fig4_stage_net_change \
         fig5_within_question_eval_gaming fig6_vea_position_in_cot fig7_first_vea_by_benchmark \
         fig8_vea_eval_gaming_split fig9_sft_dpo_data_vea fig10_eval_gaming_final_ckpts \
         fig_model_vs_data_vea fig_template_gap; do
  echo "  - $s"
  "$PY" "$s.py" >/dev/null
done
echo "done -> ${FIG_DIR:-$HERE/figures}"
