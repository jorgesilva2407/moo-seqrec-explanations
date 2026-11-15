# Written for just >= 1.40 (the [group] attribute needs >= 1.27). See `just --list`.
set shell := ["bash", "-euo", "pipefail", "-c"]

# ---- variables (override: `just tag=v0.1.0 docker-build` or `just --set tag v0.1.0 docker-build`)
image := "ghcr.io/jorgesilva2407/sasrec-ml20m"
repo_url := "https://github.com/jorgesilva2407/sasrec-ml20m"
tag := "latest"
# empty = the CLI picks <timestamp>-seed<seed>
run_name := ""
seed := "42"
# global gradient-norm clipping for train/train-bg (0 = off)
grad_clip := "0"
# "auto" (bf16 autocast) or "fp32" for train/train-bg
precision := "auto"
session := "train"
# extra uv dependency groups for docker-build, space separated (the upload group is needed for Hub uploads)
groups := "dev upload"
# Hugging Face repo (user/name): when set, train/train-bg/resume upload the run at the end (needs HF_TOKEN)
upload_to := ""
# set to "--upload-public" to create the Hub repo public (default: private)
upload_visibility := ""
smoke_users := "2000"
smoke_epochs := "2"
smoke_processed := "processed_smoke"
data_dir := env("SASREC_DATA_DIR", "/workspace/data")
out_dir := env("SASREC_OUT_DIR", "/workspace/outputs")
# where the data/outputs live on the vast.ai instance (used by pull-run)
remote_data_dir := "/workspace/data"
remote_out_dir := "/workspace/outputs"
# host folder mounted by docker-run (laptop with a GPU)
host_workspace := env("HOST_WORKSPACE", "./workspace")

# Inside the container (INSIDE_CONTAINER=1) the commands are on PATH; locally they go through uv.
inside := env("INSIDE_CONTAINER", "0")
seqrec := if inside == "1" { "seqrec" } else { "uv run seqrec" }
python := if inside == "1" { "python" } else { "uv run python" }
pytest := if inside == "1" { "python -m pytest" } else { "uv run --group dev pytest" }
run_flag := if run_name == "" { "" } else { "--run-name " + run_name }
paths_flags := "--data-dir " + data_dir + " --out-dir " + out_dir
upload_flag := if upload_to == "" { "" } else { "--upload-to " + upload_to + " " + upload_visibility }

# List all recipes
default:
    @just --list

# ------------------------------------------------------------------ local (laptop)

# Create or refresh uv.lock
[group('local')]
lock:
    uv lock

# Install the project and the dev group into .venv
[group('local')]
sync:
    uv sync --group dev

# Run the tests locally through uv (CPU)
[group('local')]
test-local:
    uv run --group dev pytest

# Fail if uv.lock is missing or older than pyproject.toml
[group('local')]
check-lock:
    #!/usr/bin/env bash
    set -euo pipefail
    [ -f uv.lock ] || { echo "uv.lock is missing: run 'just lock'"; exit 1; }
    if [ pyproject.toml -nt uv.lock ]; then
        echo "uv.lock is older than pyproject.toml: run 'just lock'"
        exit 1
    fi

# Build the Docker image locally (needs docker)
[group('local')]
docker-build: check-lock
    docker build \
        --build-arg GIT_COMMIT="$(git rev-parse --short HEAD)" \
        --build-arg DEPENDENCY_GROUPS="{{groups}}" \
        --build-arg SOURCE_URL="{{repo_url}}" \
        -t "{{image}}:{{tag}}" .

# Run the image locally with all GPUs and ./workspace mounted (needs docker + NVIDIA GPU)
[group('local')]
docker-run:
    mkdir -p "{{host_workspace}}/data" "{{host_workspace}}/outputs"
    docker run --rm -it --gpus all \
        -v "$(cd "{{host_workspace}}/data" && pwd):/workspace/data" \
        -v "$(cd "{{host_workspace}}/outputs" && pwd):/workspace/outputs" \
        "{{image}}:{{tag}}" bash

# Tag the current commit and push the tag, which triggers the CI image build (asks first)
[group('local')]
release version:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! [[ "{{version}}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "version must look like v0.1.0"; exit 1
    fi
    if [ -n "$(git status --porcelain)" ]; then
        echo "working tree is not clean: commit or stash first"; exit 1
    fi
    if git rev-parse -q --verify "refs/tags/{{version}}" >/dev/null; then
        echo "tag {{version}} already exists"; exit 1
    fi
    short="$(git rev-parse --short HEAD)"
    v="{{version}}"
    echo "About to create tag {{version}} on commit $short and push it to origin."
    read -r -p "Continue? [y/N] " answer
    [ "$answer" = "y" ] || { echo "aborted"; exit 1; }
    git tag "{{version}}"
    git push origin "{{version}}"
    echo "CI will publish:"
    echo "  {{image}}:${v#v}"
    echo "  {{image}}:sha-$short"

# Copy a run directory and data_meta.json from the vast.ai instance (needs ssh access): just pull-run HOST PORT NAME
[group('local')]
pull-run host port name:
    mkdir -p "pulled/{{name}}"
    rsync -avz -e "ssh -p {{port}}" "root@{{host}}:{{remote_out_dir}}/runs/{{name}}/" "pulled/{{name}}/"
    rsync -avz -e "ssh -p {{port}}" "root@{{host}}:{{remote_data_dir}}/processed/data_meta.json" "pulled/{{name}}/data_meta.json"

# Print the ghcr package URL and how to make the image public
[group('local')]
push-image-info:
    #!/usr/bin/env bash
    set -euo pipefail
    ref="{{image}}"
    rest="${ref#ghcr.io/}"
    owner="${rest%%/*}"
    repo="${rest#*/}"
    echo "Image: {{image}}"
    echo "Package page: https://github.com/users/$owner/packages/container/$repo"
    echo "  (organisation: https://github.com/orgs/$owner/packages/container/$repo)"
    echo "Make it public: open the package page -> Package settings -> Danger Zone -> Change visibility -> Public."
    echo "Keep it private: give vast.ai registry credentials: server ghcr.io, user <github user>,"
    echo "  password = a personal access token (classic) with the read:packages scope."
    echo "Check locally: docker pull {{image}}:{{tag}}"

# ------------------------------------------------------------------ remote (GPU machine, inside the container)

# Show the GPU and check that torch sees CUDA
[group('remote')]
gpu-check:
    nvidia-smi
    {{python}} -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

# Run the test-suite (works in the container and locally)
[group('remote')]
test:
    {{pytest}}

# Download (if needed) and preprocess ML-20M into <data-dir>/processed
[group('remote')]
prepare-data:
    {{seqrec}} prepare-data {{paths_flags}}

# Preprocess a small user subset into <data-dir>/processed_smoke
[group('remote')]
prepare-data-smoke:
    {{seqrec}} prepare-data {{paths_flags}} --max-users {{smoke_users}} --processed-name {{smoke_processed}}

# Full training run in the foreground (needs CUDA)
[group('remote')]
train:
    {{seqrec}} train {{paths_flags}} --seed {{seed}} --grad-clip {{grad_clip}} --precision {{precision}} {{run_flag}} {{upload_flag}}

# Short training run on the smoke subset (needs CUDA); prints seconds per epoch
[group('remote')]
train-smoke:
    {{seqrec}} train {{paths_flags}} --seed {{seed}} --processed-name {{smoke_processed}} --epochs {{smoke_epochs}} --run-name "smoke-$(date +%Y%m%d-%H%M%S)"

# Start the full training in a detached tmux session and tee the output to the run directory
[group('remote')]
train-bg:
    #!/usr/bin/env bash
    set -euo pipefail
    if tmux has-session -t "{{session}}" 2>/dev/null; then
        echo "tmux session '{{session}}' already exists (just attach, or kill it first)"; exit 1
    fi
    name="{{run_name}}"
    if [ -z "$name" ]; then name="$(date +%Y%m%d-%H%M%S)-seed{{seed}}"; fi
    run_dir="{{out_dir}}/runs/$name"
    mkdir -p "$run_dir"
    # an already running tmux server would not pass HF_TOKEN on, so hand it over explicitly
    env_args=()
    if [ -n "${HF_TOKEN:-}" ]; then env_args=(-e "HF_TOKEN=$HF_TOKEN"); fi
    tmux new-session -d -s "{{session}}" "${env_args[@]}" bash -c \
        "set -o pipefail; {{seqrec}} train {{paths_flags}} --seed {{seed}} --grad-clip {{grad_clip}} --precision {{precision}} {{upload_flag}} --run-name '$name' 2>&1 | tee -a '$run_dir/console.log'; echo; echo 'training process ended'; exec bash"
    echo "started run '$name' in tmux session '{{session}}'"
    echo "attach:      just attach     (detach with Ctrl-b d)"
    echo "follow log:  just logs       (or: tail -f $run_dir/train.log)"
    echo "resume later: just resume $name"

# Attach to the training tmux session
[group('remote')]
attach:
    tmux attach -t "{{session}}"

# Follow the newest run's train.log
[group('remote')]
logs:
    #!/usr/bin/env bash
    set -euo pipefail
    latest="$(ls -t "{{out_dir}}"/runs/*/train.log 2>/dev/null | head -n 1 || true)"
    [ -n "$latest" ] || { echo "no train.log under {{out_dir}}/runs"; exit 1; }
    echo "following $latest"
    tail -n 50 -f "$latest"

# Resume an interrupted run from its last.pt (needs CUDA): just resume NAME
[group('remote')]
resume name:
    {{seqrec}} train {{paths_flags}} --resume --run-name "{{name}}" {{upload_flag}}

# Upload a finished run to the Hugging Face Hub (network; needs HF_TOKEN): just upload NAME user/repo
[group('remote')]
upload name repo:
    {{seqrec}} upload {{paths_flags}} --run-name "{{name}}" --repo-id "{{repo}}" {{upload_visibility}}

# Evaluate a checkpoint on the processed data: just evaluate /workspace/outputs/runs/NAME/best.pt
[group('remote')]
evaluate checkpoint:
    {{seqrec}} evaluate {{paths_flags}} --checkpoint "{{checkpoint}}"

# Delete only smoke-run outputs (runs/smoke-* and processed_smoke), after confirmation
[group('remote')]
clean-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s nullglob
    targets=("{{out_dir}}"/runs/smoke-* "{{data_dir}}/{{smoke_processed}}")
    existing=()
    for t in "${targets[@]}"; do [ -e "$t" ] && existing+=("$t"); done
    if [ "${#existing[@]}" -eq 0 ]; then echo "nothing to remove"; exit 0; fi
    echo "Will remove:"
    printf '  %s\n' "${existing[@]}"
    read -r -p "Delete these? [y/N] " answer
    [ "$answer" = "y" ] || { echo "aborted"; exit 1; }
    rm -rf -- "${existing[@]}"
    echo "removed"
