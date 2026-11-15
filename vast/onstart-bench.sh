#!/bin/bash
# vast.ai onstart script: prepare the data and run the verifier benchmark without an SSH session.
# Used as the template's onstart command (`just vast-template-bench`). Progress goes to
# /workspace/outputs/onstart.log; the last line is "[bench] DONE" or "[bench] FAILED ...".
#
# With HF_TOKEN (or SASREC_HF_TOKEN) in the instance environment the checkpoint comes from the Hub
# and the results are uploaded to the results dataset. Without it the script waits for a
# checkpoint copied to /workspace/data/best.pt and uploads nothing.

RESULTS_REPO="${SASREC_RESULTS_REPO:-AbyssWanderer/sasrec-ml20m-explanations}"
BENCH_ARGS="${SASREC_BENCH_ARGS:---per-stratum 10}"

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

bench() {
    echo "[bench] start $(date -u +%FT%TZ) commit ${GIT_COMMIT:-unknown}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "[bench] cores $(nproc), ram $(free -g | awk 'NR==2 {print $2}') GB"

    if [ ! -f /workspace/data/processed/data_meta.json ]; then
        just prepare-data || { echo "[bench] FAILED prepare-data"; return 1; }
    fi

    checkpoint=""
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "[bench] no HF_TOKEN: waiting up to 1 h for /workspace/data/best.pt"
        for _ in $(seq 360); do [ -f /workspace/data/best.pt ] && break; sleep 10; done
        [ -f /workspace/data/best.pt ] || { echo "[bench] FAILED no checkpoint"; return 1; }
        checkpoint="--checkpoint /workspace/data/best.pt"
    fi

    # peak GPU memory, sampled while the benchmark runs
    ( while sleep 1; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done > /tmp/gpu_mem.log ) &
    sampler=$!
    # shellcheck disable=SC2086
    just explain-bench $checkpoint $BENCH_ARGS
    status=$?
    kill "$sampler" 2>/dev/null
    echo "[bench] peak GPU memory $(sort -n /tmp/gpu_mem.log | tail -n 1) MiB"
    [ "$status" -eq 0 ] || { echo "[bench] FAILED explain-bench"; return 1; }

    if [ -n "${HF_TOKEN:-}" ]; then
        gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 | tr ' ' '_')"
        python -c "
import sys
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='/workspace/outputs/explanations/bench', repo_id=sys.argv[1],
                      repo_type='dataset', path_in_repo='bench/' + sys.argv[2])
" "$RESULTS_REPO" "$gpu" && echo "[bench] uploaded to $RESULTS_REPO" || echo "[bench] WARNING upload failed"
    fi
    echo "[bench] DONE $(date -u +%FT%TZ)"
}

bench > /workspace/outputs/onstart.log 2>&1 &
