#!/usr/bin/env bash
# Run token_ref_1ref experiments across multiple base models and Pile subsets
# Usage: bash scripts/token_ref_1ref.sh

set -euo pipefail
IFS=$'\n\t'

# Resolve repository root (this script lives in scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Activate venv if present
if [[ -d "venv" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

CONFIG="configs/token_ref_1ref_mink.json"
NGRAM=13

# Models and subsets to iterate
MODELS=(
  # "pythia-160m"
  "pythia-1.4b"
  # "pythia-2.8b"
  # "pythia-6.9b"
)

SUBSETS=(
  "wikipedia_(en)"
  "github"
  "pile_cc"
  "pubmed_central"
  "arxiv"
  "dm_mathematics"
  "hackernews"
)

# Optional: create a simple log directory
LOG_DIR="logs/token_ref_1ref_mink"
mkdir -p "${LOG_DIR}"

for model in "${MODELS[@]}"; do
  BASE_MODEL="EleutherAI/${model}-deduped"
  for subset in "${SUBSETS[@]}"; do
    SPECIFIC_SOURCE="${subset}_ngram_${NGRAM}_0.8_truncated"
    echo "[token_ref_1ref_mink] Running base_model=${BASE_MODEL} subset=${subset}"
    # Each run writes a short log
    SAFE_MODEL="${model//\//_}"
    SAFE_SUBSET="${subset//\//_}"
    LOG_FILE="${LOG_DIR}/${SAFE_MODEL}__${SAFE_SUBSET}.log"

    python run.py \
      --config "${CONFIG}" \
      --base_model "${BASE_MODEL}" \
      --specific_source "${SPECIFIC_SOURCE}" \
      2>&1 | tee "${LOG_FILE}"
  done
done
