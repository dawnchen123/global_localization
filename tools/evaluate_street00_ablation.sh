#!/usr/bin/env bash
set -euo pipefail

WS="${WS:-/home/dawn/workspace/fast_livo2_global_ws}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/hybrid_street00_ablation}"
GT="${GT:-/home/dawn/文档/phd_exp/street00/street00_trajectory.csv}"
EVO_BIN="${EVO_BIN:-/home/dawn/文档/phd_exp/evo_env/bin}"
EVO_HOME="${EVO_HOME:-/home/dawn/文档/phd_exp/evo_home}"
GT_TIME_SHIFT="${GT_TIME_SHIFT:-27.0}"
EVAL_DIR="${OUTPUT_ROOT}/evaluation"

ESTIMATES=(
  "${OUTPUT_ROOT}/lio/frontend.csv"
  "${OUTPUT_ROOT}/lio_visual/frontend.csv"
  "${OUTPUT_ROOT}/lio_visual_sam3/frontend.csv"
  "${OUTPUT_ROOT}/full_graph/graph.csv"
)
NAMES=(lio lio_visual lio_visual_sam3 full_graph)

for estimate in "${ESTIMATES[@]}"; do
  if [[ ! -s "${estimate}" ]]; then
    echo "Missing trajectory: ${estimate}" >&2
    exit 2
  fi
done

python3 "${WS}/src/fast_livo2_global_localization/tools/evo_local_ned_eval.py" \
  --dataset_dir "$(dirname "${GT}")" \
  --gt "${GT}" \
  --est "${ESTIMATES[@]}" \
  --names "${NAMES[@]}" \
  --out_dir "${EVAL_DIR}" \
  --evo_bin "${EVO_BIN}" \
  --evo_home "${EVO_HOME}" \
  --time_mode relative \
  --gt_time_shift "${GT_TIME_SHIFT}" \
  --gt_time_cluster all \
  --est_time_cluster auto \
  --max_time_diff 0.05 \
  --align se3 \
  --run_rpe \
  --rpe_delta 1.0 \
  --rpe_delta_unit m

python3 "${WS}/src/fast_livo2_global_localization/tools/format_ablation_summary.py" \
  "${EVAL_DIR}/summary.json"
