#!/bin/bash
# Replay every method, clean and under each photometric stress, into results/.
set -u
cd "$(dirname "$0")"
R=/home/josh/Documents/repos/research/hitl-pmp-results/tidybot-real/analysis/2026-10-20-pilot/bin_detector/results
PY=~/miniconda3/envs/hitl-pmp/bin/python
for p in ${PERTURBS:-none dim bright desat warm}; do
  for m in ${METHODS:-sat130 sat150 edge learn contrast}; do
    out=$R/${m}_${p}.jsonl
    [ -s "$out" ] && [ "$(wc -l < "$out")" -ge 270 ] && continue
    model=""; [ "$m" = learn ] && model=$R/learn_model.pkl
    PERTURB=$p PYTHONPATH=../../src:. OMP_NUM_THREADS=1 $PY evaluate.py $m "$out" "$model" ${WORKERS:-10} 2>&1 | grep -v "pybullet build\|moov"
    echo "$(date +%H:%M) done $m $p"
  done
done
