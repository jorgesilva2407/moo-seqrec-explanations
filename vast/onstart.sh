#!/bin/bash
# vast.ai onstart script: prepare the data and run one job, with no SSH session needed.
# Installed as the template's onstart command by `just vast-template <image-tag>`.
#
# The jobs are `just` recipes with their arguments, run in order, taken from the instance
# environment (default: the verifier benchmark). vast.ai drops variables whose value has spaces,
# so write "+" for a space and "," between jobs:
#   --env '-e SASREC_LABEL=main-0 -e SASREC_JOB=explain-run+--stage+main+--shard+0/4'
#   SASREC_JOB=explain-run+--stage+tune-a+--pop-size+50,explain-run+--stage+tune-b+--pop-size+100
# HF_TOKEN (or SASREC_HF_TOKEN) must be in the environment: the checkpoint comes from the Hub and
# results go to the dataset SASREC_RESULTS_REPO. Progress is in /workspace/outputs/onstart.log,
# which ends with "[job] DONE" or "[job] FAILED ..." and is uploaded to logs/ in the dataset
# (named after SASREC_LABEL).

RESULTS_REPO="${SASREC_RESULTS_REPO:-AbyssWanderer/sasrec-ml20m-explanations}"
JOB="${SASREC_JOB:-explain-bench+--per-stratum+10}"
JOB="${JOB//+/ }"
LOG=/workspace/outputs/onstart.log

mkdir -p /workspace/data /workspace/outputs
T="${SASREC_HF_TOKEN:-${HF_TOKEN:-}}"
if [ -n "$T" ]; then
    sed -i "/^HF_TOKEN=/d" /etc/environment
    printf "HF_TOKEN=%s\n" "$T" >> /etc/environment
    printf "export HF_TOKEN=%q\n" "$T" > /etc/profile.d/hf_token.sh
    chmod 600 /etc/profile.d/hf_token.sh
    export HF_TOKEN="$T"
fi
. /etc/profile.d/seqrec.sh
cd /opt/seqrec || exit 1

job() {
    echo "[job] start $(date -u +%FT%TZ) commit ${GIT_COMMIT:-unknown}: $JOB"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "[job] cores $(nproc), ram $(free -g | awk 'NR==2 {print $2}') GB"
    [ -n "${HF_TOKEN:-}" ] || { echo "[job] FAILED no HF_TOKEN in the instance environment"; return 1; }
    if [ ! -f /workspace/data/processed/data_meta.json ]; then
        just prepare-data || { echo "[job] FAILED prepare-data"; return 1; }
    fi
    IFS=',' read -ra steps <<< "$JOB"
    for step in "${steps[@]}"; do
        echo "[job] step $(date -u +%FT%TZ): $step"
        # shellcheck disable=SC2086
        just $step --results-repo "$RESULTS_REPO" || { echo "[job] FAILED $step"; return 1; }
    done
    echo "[job] DONE $(date -u +%FT%TZ)"
}

upload_log() {
    name="$(date -u +%Y%m%d-%H%M%S)-${SASREC_LABEL:-job}-${CONTAINER_ID:-$(hostname)}.log"
    python -c "
import sys
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj=sys.argv[1], path_in_repo='logs/' + sys.argv[2], repo_id=sys.argv[3], repo_type='dataset')
" "$LOG" "$name" "$RESULTS_REPO" || true
}

( job > "$LOG" 2>&1; upload_log >> "$LOG" 2>&1 ) &
