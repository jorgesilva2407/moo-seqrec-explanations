#!/bin/bash
# Round 2, step 1b: what was run by hand on the Z0 machine (image sha-b185f3d lacked `six`).
# Equivalent to Z0_ENV in 01-first-machines.sh on image sha-d7c46c6 or later.
cd /opt/seqrec
uv pip install --python /opt/seqrec/.venv/bin/python six >/dev/null 2>&1
LOG=/workspace/outputs/onstart.log
ZR=AbyssWanderer/moo-seqrec-explanations-zvuk
(
  echo "[job] manual restart $(date -u +%FT%TZ): six installed (kaggle 2.2.5 lacks the dependency)"
  just prepare-data --dataset zvuk --processed-name processed_zvuk --max-items 50000 --max-users 100000 &&
  just push-processed processed_zvuk $ZR &&
  just train-with --processed-name processed_zvuk --hidden-size 256 --num-layers 3 --dropout 0.1 --precision fp32 --grad-clip 1.0 --epochs 1 --val-subset 5000 --run-name smoke-h256-l3 &&
  echo "[job] DONE $(date -u +%FT%TZ)" || echo "[job] FAILED manual run"
  python -c "
import sys
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='logs/z0prep-manual.log', repo_id=sys.argv[2], repo_type='dataset')
" "$LOG" "$ZR"
) >> "$LOG" 2>&1
