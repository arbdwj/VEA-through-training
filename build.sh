#!/usr/bin/env bash
# Regenerate all figures from bundled data, then compile the PDF.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"   # set PY=/path/to/venv/python if needed (needs numpy pandas matplotlib)

echo "[1/2] regenerating figures..."
cd "$HERE/scripts"
for s in fig1_pretrain_vea fig2_rlvr_vea_benchmarks fig3_stage_connected \
         fig4_stage_net_change fig5_within_question_eval_gaming; do
  echo "  - $s"
  "$PY" "$s.py" >/dev/null
done

echo "[2/2] compiling vea_training.pdf..."
cd "$HERE"
pdflatex -interaction=nonstopmode -halt-on-error vea_training.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error vea_training.tex >/dev/null
echo "done -> $HERE/vea_training.pdf"
