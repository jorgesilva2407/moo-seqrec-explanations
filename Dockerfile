# One image for the whole project (training now, other commands later).
# Build:  docker build --build-arg DEPENDENCY_GROUPS="dev" -t seqrec .   (default: core only)

# CUDA version must match the PyTorch index in pyproject.toml (cu126 <-> 12.6).
# PyTorch wheels bundle the CUDA libraries, so the lightweight "base" flavor is enough;
# only the host driver must support CUDA 12.6.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-base-ubuntu24.04
ARG UV_VERSION=0.8.15

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${CUDA_IMAGE}

ARG JUST_VERSION=1.40.0
ARG PYTHON_VERSION=3.12
# Space separated uv dependency groups to install, e.g. "dev" (empty = core only).
ARG DEPENDENCY_GROUPS=""

# Packages for a remote workflow. No openssh-server: vast.ai's SSH launch mode injects its own
# SSH access (TODO(verify) in the vast.ai docs); openssh-client is for git over ssh.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git tmux rsync curl unzip ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Pinned uv binary; uv downloads and manages Python itself (no system Python).
COPY --from=uv /uv /uvx /usr/local/bin/
ENV UV_PYTHON=${PYTHON_VERSION} \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_TOOL_DIR=/opt/uv-tools

# just from PyPI (rust-just ships the prebuilt binary): same pinned install path as everything
# else here, no extra download URLs or architecture handling. Binary lands in /usr/local/bin.
RUN uv tool install "rust-just==${JUST_VERSION}"

WORKDIR /opt/seqrec
ENV UV_PROJECT_ENVIRONMENT=/opt/seqrec/.venv \
    PATH="/opt/seqrec/.venv/bin:${PATH}"

# Dependencies first (cached layer). --frozen fails the build if uv.lock is missing or stale.
COPY pyproject.toml uv.lock ./
RUN set -eu; flags=""; for g in ${DEPENDENCY_GROUPS}; do flags="${flags} --group ${g}"; done; \
    uv sync --frozen --no-install-project ${flags}

# Then the project itself.
COPY src ./src
COPY tests ./tests
COPY justfile ./justfile
RUN set -eu; flags=""; for g in ${DEPENDENCY_GROUPS}; do flags="${flags} --group ${g}"; done; \
    uv sync --frozen ${flags}

ENV INSIDE_CONTAINER=1 \
    SASREC_DATA_DIR=/workspace/data \
    SASREC_OUT_DIR=/workspace/outputs \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTHONUNBUFFERED=1

# vast.ai SSH sessions do not inherit the image's ENV, so persist what the justfile needs for
# login shells (profile.d) and for non-login ones (/etc/environment); land in /opt/seqrec.
RUN { \
      echo 'export INSIDE_CONTAINER=1'; \
      echo 'export SASREC_DATA_DIR=/workspace/data'; \
      echo 'export SASREC_OUT_DIR=/workspace/outputs'; \
      echo 'export PYTHONUNBUFFERED=1'; \
      echo 'export PATH=/opt/seqrec/.venv/bin:$PATH'; \
    } > /etc/profile.d/seqrec.sh \
    && { \
      echo 'INSIDE_CONTAINER=1'; \
      echo 'SASREC_DATA_DIR=/workspace/data'; \
      echo 'SASREC_OUT_DIR=/workspace/outputs'; \
      echo 'PYTHONUNBUFFERED=1'; \
    } >> /etc/environment \
    && echo '[ -d /opt/seqrec ] && cd /opt/seqrec' >> /root/.bashrc \
    && mkdir -p /workspace/data /workspace/outputs

# Placed last so a new commit does not invalidate the dependency layers.
ARG GIT_COMMIT=unknown
ARG SOURCE_URL=""
ENV GIT_COMMIT=${GIT_COMMIT}
LABEL org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.description="SASRec (full cross-entropy) on MovieLens-20M: training and evaluation" \
      org.opencontainers.image.revision="${GIT_COMMIT}"

# Runs as root (vast.ai expects it). Idle by default: SSH in and use `just` inside tmux.
CMD ["sleep", "infinity"]
