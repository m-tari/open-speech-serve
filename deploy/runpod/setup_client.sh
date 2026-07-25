#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/open-speech-serve}"

apt-get update
apt-get install -y --no-install-recommends ffmpeg libsndfile1 git

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone https://github.com/m-tari/open-speech-serve.git "${REPO_DIR}"
fi

python -m pip install --upgrade pip
python -m pip install httpx jiwer matplotlib pyyaml soundfile
python -m pip install --no-deps -e "${REPO_DIR}"

mkdir -p "${REPO_DIR}/data" "${REPO_DIR}/results"
echo "Benchmark client ready in ${REPO_DIR}"
