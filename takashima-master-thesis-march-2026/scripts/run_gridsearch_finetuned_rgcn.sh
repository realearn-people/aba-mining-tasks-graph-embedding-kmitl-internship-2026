#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG="config/robust_experiment.grid.yaml"
ROOT_ID="exp_37_20251123_FinetunedBertRgcnMlp_gridsearch"

echo "Using config: ${CONFIG}"
echo "Experiment root: ${ROOT_ID}"

source .venv/bin/activate

python -u src/experiments/run_named_models.py \
  --config "${CONFIG}" \
  --only-model FinetunedBertRgcnMlp \
  --search-strategy grid \
  --sweep-config "data/training_results/${ROOT_ID}/sweep_finetuned_rgcn.json" \
  --experiment-id "${ROOT_ID}/FinetunedBertRgcnMlp"

echo "FinetunedBertRgcnMlp gridsearch completed."


