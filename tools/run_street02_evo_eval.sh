#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/dawn/document/phd_exp"
SCRIPT="${BASE_DIR}/evo_tools/evo_local_ned_eval.py"
DATASET="${1:-street00}"
GT_TIME_SHIFT="${GT_TIME_SHIFT:-27.0}"
ALIGN="${ALIGN:-se3}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/${DATASET}/evo_eval_shift${GT_TIME_SHIFT}_${ALIGN}}"

python3 "${SCRIPT}" \
  --dataset_dir "${BASE_DIR}/${DATASET}" \
  --gt "${BASE_DIR}/${DATASET}/${DATASET}_trajectory.csv" \
  --out_dir "${OUT_DIR}" \
  --evo_bin "${BASE_DIR}/evo_env/bin" \
  --evo_home "${BASE_DIR}/evo_home" \
  --time_mode relative \
  --gt_time_shift "${GT_TIME_SHIFT}" \
  --gt_time_cluster all \
  --est_time_cluster auto \
  --max_time_diff 0.05 \
  --align "${ALIGN}" \
  --run_rpe
