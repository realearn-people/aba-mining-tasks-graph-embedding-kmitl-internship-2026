#!/usr/bin/env bash
set -euo pipefail

# 実験ルート（既存ディレクトリを使用）
ROOT_DIR="exp_20251114_gridsearch_models_room_staff"
CONFIG="config/robust_experiment.grid.yaml"

echo "Using config: ${CONFIG}"
echo "Experiment root: ${ROOT_DIR}"

# 1) FreezedBertRgcnMlp
python -u src/experiments/run_named_models.py \
  --config "${CONFIG}" \
  --only-model FreezedBertRgcnMlp \
  --search-strategy grid \
  --sweep-config "data/training_results/exp_23_rgcn_FreezedBertRgcnMlp_sweep/sweep_rgcn.json" \
  --experiment-id "${ROOT_DIR}/FreezedBertRgcnMlp"

# 2) FreezedBertMlp
python -u src/experiments/run_named_models.py \
  --config "${CONFIG}" \
  --only-model FreezedBertMlp \
  --search-strategy grid \
  --sweep-config "data/training_results/exp_24_1105_FreezedBertMlp/sweep_freezed_bert_mlp.json" \
  --experiment-id "${ROOT_DIR}/FreezedBertMlp"

# 3) FinetunedBertMlp
python -u src/experiments/run_named_models.py \
  --config "${CONFIG}" \
  --only-model FinetunedBertMlp \
  --search-strategy grid \
  --sweep-config "data/training_results/exp_21_1103_FinetunedBertMlp/sweep_probe.json" \
  --experiment-id "${ROOT_DIR}/FinetunedBertMlp"

# 4) FinetunedBertCosSim
python -u src/experiments/run_named_models.py \
  --config "${CONFIG}" \
  --only-model FinetunedBertCosSim \
  --search-strategy grid \
  --sweep-config "data/training_results/${ROOT_DIR}/sweep_cossim.json" \
  --experiment-id "${ROOT_DIR}/FinetunedBertCosSim"

# 5) TfidfLr
python -u src/experiments/run_named_models.py \
  --config "${CONFIG}" \
  --only-model TfidfLr \
  --search-strategy grid \
  --sweep-config "data/training_results/${ROOT_DIR}/sweep_tfidf_lr.json" \
  --experiment-id "${ROOT_DIR}/TfidfLr"

# 後処理: 集約可視化
python -u scripts/aggregate_best_bars.py \
  --root "data/training_results/${ROOT_DIR}"

echo "All grid searches completed."


