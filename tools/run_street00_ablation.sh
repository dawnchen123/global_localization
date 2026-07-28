#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
case "${MODE}" in
  lio|lio_visual|lio_visual_sam3|full_graph) ;;
  *)
    echo "Usage: $0 {lio|lio_visual|lio_visual_sam3|full_graph}" >&2
    exit 2
    ;;
esac

case "${MODE}" in
  lio) DEFAULT_ROS_PORT=11511 ;;
  lio_visual) DEFAULT_ROS_PORT=11512 ;;
  lio_visual_sam3) DEFAULT_ROS_PORT=11513 ;;
  full_graph) DEFAULT_ROS_PORT=11514 ;;
esac

WS="${WS:-/home/dawn/workspace/fast_livo2_global_ws}"
BAG="${BAG:-/media/dawn/8fa0cb2a-6f7b-45d7-85ec-ac18e44d2f6b/dataset/slam/i2Nav-Robot/street00/street00.bag}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/hybrid_street00_ablation}"
DURATION="${DURATION:-60}"
START="${START:-0}"
RATE="${RATE:-1.0}"
POST_RUN_WAIT="${POST_RUN_WAIT:-20}"
SETUP_FILE="${SETUP_FILE:-${WS}/install_hybrid/setup.bash}"
SAM3_PYTHON="${SAM3_PYTHON:-/home/dawn/.local/share/mamba/envs/sam3/bin/python}"
SAM3_ROOT="${SAM3_ROOT:-/home/dawn/software/sam3}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-${SAM3_ROOT}/checkpoints/sam3.pt}"
SAM3_STARTUP_TIMEOUT="${SAM3_STARTUP_TIMEOUT:-180}"
RESULT_DIR="${OUTPUT_ROOT}/${MODE}"
QUEUE_DIR="${RESULT_DIR}/sam3_queue"
ROS_MASTER_PORT="${ROS_MASTER_PORT:-${DEFAULT_ROS_PORT}}"
export ROS_MASTER_URI="http://127.0.0.1:${ROS_MASTER_PORT}"

if [[ -e "${RESULT_DIR}/frontend.csv" || -e "${RESULT_DIR}/graph.csv" ]]; then
  echo "Refusing to overwrite an existing run: ${RESULT_DIR}" >&2
  exit 5
fi
mkdir -p "${RESULT_DIR}" "${RESULT_DIR}/ros_home"
export ROS_HOME="${RESULT_DIR}/ros_home"
if [[ ! -r "${SETUP_FILE}" ]]; then
  echo "Workspace setup file not found: ${SETUP_FILE}" >&2
  exit 6
fi
source "${SETUP_FILE}"

SUBSCRIBE_CAMERA=false
ENABLE_VISUAL=false
# Leave these unset until the selected mode establishes its default.  This
# lets a full-graph replay isolate visual loop closure from SAM3 by passing
# ENABLE_SAM3=false, instead of forcing the GPU service into every graph test.
ENABLE_SAM3="${ENABLE_SAM3:-}"
ENABLE_GRAPH="${ENABLE_GRAPH:-}"
SUBSCRIBE_WHEEL="${SUBSCRIBE_WHEEL:-true}"
# Leave graph wheel ownership to the loaded configuration unless an ablation
# explicitly passes true/false. This mirrors the profile-preserving behavior
# for the sequential ground-Z experiment switch.
ENABLE_WHEEL_FACTORS="${ENABLE_WHEEL_FACTORS:-}"
ENABLE_GENERIC_LOOPS="${ENABLE_GENERIC_LOOPS:-false}"
ENABLE_XY_LOOPS="${ENABLE_XY_LOOPS:-${ENABLE_GENERIC_LOOPS}}"
ENABLE_Z_LOOPS="${ENABLE_Z_LOOPS:-${ENABLE_GENERIC_LOOPS}}"
ENABLE_SEQUENTIAL_GROUND_Z="${ENABLE_SEQUENTIAL_GROUND_Z:-}"
if [[ -z "${ENABLE_SEQUENTIAL_GROUND_Z}" && "${ENABLE_GENERIC_LOOPS}" == true ]]; then
  ENABLE_SEQUENTIAL_GROUND_Z=true
fi
SUBSCRIBE_CAMERA_GRAPH="${SUBSCRIBE_CAMERA_GRAPH:-}"
PROCESS_REGISTERED_CLOUDS="${PROCESS_REGISTERED_CLOUDS:-}"
VISUAL_FUSION_PROFILE="${VISUAL_FUSION_PROFILE:-}"
VISUAL_OBSERVATION_ONLY="${VISUAL_OBSERVATION_ONLY:-}"
VISUAL_AUTO_Z_WHEN_OBSERVABLE="${VISUAL_AUTO_Z_WHEN_OBSERVABLE:-}"
FRONTEND_TUNING_CONFIG="${FRONTEND_TUNING_CONFIG:-}"
GRAPH_TUNING_CONFIG="${GRAPH_TUNING_CONFIG:-}"
if [[ "${MODE}" != "lio" ]]; then
  SUBSCRIBE_CAMERA=true
  ENABLE_VISUAL=true
  # lio_visual and later ablations are intended to measure the direct
  # LiDAR-depth visual ESKF, not merely collect KLT/PnP diagnostics.
  VISUAL_FUSION_PROFILE="${VISUAL_FUSION_PROFILE:-true}"
  VISUAL_OBSERVATION_ONLY="${VISUAL_OBSERVATION_ONLY:-false}"
else
  VISUAL_FUSION_PROFILE="${VISUAL_FUSION_PROFILE:-false}"
  VISUAL_OBSERVATION_ONLY="${VISUAL_OBSERVATION_ONLY:-true}"
fi
if [[ "${MODE}" == "lio_visual_sam3" || "${MODE}" == "full_graph" ]]; then
  ENABLE_SAM3="${ENABLE_SAM3:-true}"
fi
if [[ "${MODE}" == "full_graph" ]]; then
  ENABLE_GRAPH="${ENABLE_GRAPH:-true}"
fi
: "${ENABLE_SAM3:=false}"
: "${ENABLE_GRAPH:=false}"
if [[ -z "${SUBSCRIBE_CAMERA_GRAPH}" ]]; then
  SUBSCRIBE_CAMERA_GRAPH="${ENABLE_GRAPH}"
fi
if [[ -z "${PROCESS_REGISTERED_CLOUDS}" ]]; then
  # SAM3 semantic association and graph factors consume registered clouds.
  # Frontend-only modes have no such consumer, so do not create a lagging
  # backend keyframe queue during their reference runs.
  if [[ "${ENABLE_GRAPH}" == true || "${ENABLE_SAM3}" == true ]]; then
    PROCESS_REGISTERED_CLOUDS=true
  else
    PROCESS_REGISTERED_CLOUDS=false
  fi
fi
# Keep the semantic factor groups independently switchable for controlled
# ablations.  Defaults preserve the selected mode's existing behavior.
ENABLE_SEMANTIC_OBSERVATION_FACTORS="${ENABLE_SEMANTIC_OBSERVATION_FACTORS:-${ENABLE_GRAPH}}"
ENABLE_SEMANTIC_XY_FACTORS="${ENABLE_SEMANTIC_XY_FACTORS:-${ENABLE_GRAPH}}"
ENABLE_SEMANTIC_Z_FACTORS="${ENABLE_SEMANTIC_Z_FACTORS:-${ENABLE_GRAPH}}"

CORE_PID=""
SERVICE_PID=""
LAUNCH_PID=""
STATUS_ECHO_PID=""
STATS_ECHO_PID=""
cleanup() {
  set +e
  [[ -n "${STATUS_ECHO_PID}" ]] && kill -INT "${STATUS_ECHO_PID}" 2>/dev/null
  [[ -n "${STATS_ECHO_PID}" ]] && kill -INT "${STATS_ECHO_PID}" 2>/dev/null
  [[ -n "${LAUNCH_PID}" ]] && kill -INT "${LAUNCH_PID}" 2>/dev/null
  [[ -n "${SERVICE_PID}" ]] && kill -TERM "${SERVICE_PID}" 2>/dev/null
  [[ -n "${CORE_PID}" ]] && kill -INT "${CORE_PID}" 2>/dev/null
  [[ -n "${LAUNCH_PID}" ]] && wait "${LAUNCH_PID}" 2>/dev/null
  [[ -n "${STATUS_ECHO_PID}" ]] && wait "${STATUS_ECHO_PID}" 2>/dev/null
  [[ -n "${STATS_ECHO_PID}" ]] && wait "${STATS_ECHO_PID}" 2>/dev/null
  if [[ -n "${SERVICE_PID}" ]]; then
    for _ in $(seq 1 20); do
      kill -0 "${SERVICE_PID}" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "${SERVICE_PID}" 2>/dev/null || true
    wait "${SERVICE_PID}" 2>/dev/null
  fi
  [[ -n "${CORE_PID}" ]] && wait "${CORE_PID}" 2>/dev/null
}
trap cleanup EXIT INT TERM

roscore -p "${ROS_MASTER_PORT}" >"${RESULT_DIR}/roscore.log" 2>&1 &
CORE_PID=$!
sleep 2
rosparam set use_sim_time true

if [[ "${ENABLE_SAM3}" == true ]]; then
  PYTHONUNBUFFERED=1 "${SAM3_PYTHON}" \
    "${WS}/src/fast_livo2_global_localization/scripts/sam3_image_mask_service.py" \
    --queue_dir "${QUEUE_DIR}" \
    --sam3_root "${SAM3_ROOT}" \
    --sam3_checkpoint "${SAM3_CHECKPOINT}" \
    --device cuda --sam3_dtype bf16 --max_batch 1 \
    >"${RESULT_DIR}/sam3_service.log" 2>&1 &
  SERVICE_PID=$!
  for _ in $(seq 1 "${SAM3_STARTUP_TIMEOUT}"); do
    if grep -q "image mask service ready" "${RESULT_DIR}/sam3_service.log"; then
      break
    fi
    if ! kill -0 "${SERVICE_PID}" 2>/dev/null; then
      echo "SAM3 service exited during startup; see ${RESULT_DIR}/sam3_service.log" >&2
      exit 3
    fi
    sleep 1
  done
  if ! grep -q "image mask service ready" "${RESULT_DIR}/sam3_service.log"; then
    echo "Timed out waiting for SAM3 initialization" >&2
    exit 4
  fi
fi

LAUNCH_ARGS=(
  rviz:=false
  "visual_fusion_profile:=${VISUAL_FUSION_PROFILE}"
  "visual_auto_z_when_observable:=${VISUAL_AUTO_Z_WHEN_OBSERVABLE}"
  "frontend_tuning_config:=${FRONTEND_TUNING_CONFIG}"
  "graph_tuning_config:=${GRAPH_TUNING_CONFIG}"
  "subscribe_wheel:=${SUBSCRIBE_WHEEL}"
  "subscribe_camera:=${SUBSCRIBE_CAMERA}"
  "subscribe_camera_graph:=${SUBSCRIBE_CAMERA_GRAPH}"
  "process_registered_clouds:=${PROCESS_REGISTERED_CLOUDS}"
  "enable_visual_frontend:=${ENABLE_VISUAL}"
  "visual_observation_only:=${VISUAL_OBSERVATION_ONLY}"
  enable_visual_rotation_factors:=false
  enable_visual_translation_factors:=false
  "enable_visual_loop_factors:=${ENABLE_GRAPH}"
  "enable_sam3_semantics:=${ENABLE_SAM3}"
  "enable_xy_loops:=${ENABLE_XY_LOOPS}"
  "enable_z_loops:=${ENABLE_Z_LOOPS}"
  "enable_semantic_observation_factors:=${ENABLE_SEMANTIC_OBSERVATION_FACTORS}"
  "enable_semantic_observation_xy_factors:=${ENABLE_SEMANTIC_XY_FACTORS}"
  "enable_semantic_observation_z_factors:=${ENABLE_SEMANTIC_Z_FACTORS}"
  "sam3_queue_dir:=${QUEUE_DIR}"
  "frontend_trajectory_save_path:=${RESULT_DIR}/frontend.csv"
  "trajectory_save_path:=${RESULT_DIR}/graph.csv"
)
if [[ -n "${ENABLE_SEQUENTIAL_GROUND_Z}" ]]; then
  LAUNCH_ARGS+=("enable_sequential_ground_z:=${ENABLE_SEQUENTIAL_GROUND_Z}")
fi
if [[ -n "${ENABLE_WHEEL_FACTORS}" ]]; then
  LAUNCH_ARGS+=("enable_wheel_factors:=${ENABLE_WHEEL_FACTORS}")
fi
roslaunch fast_livo2_global_localization hybrid_localization_hesai.launch \
  "${LAUNCH_ARGS[@]}" >"${RESULT_DIR}/launch.log" 2>&1 &
LAUNCH_PID=$!
sleep 4
rostopic echo /hybrid/status >"${RESULT_DIR}/frontend_status_stream.txt" 2>&1 &
STATUS_ECHO_PID=$!
rostopic echo /hybrid/semantic_graph/stats >"${RESULT_DIR}/graph_stats_stream.txt" 2>&1 &
STATS_ECHO_PID=$!

TOPICS=(/adi/adis16465/imu /hesai/at128/points)
if [[ "${SUBSCRIBE_WHEEL}" == true ]]; then
  TOPICS+=(/insprobe/ranger/odometer)
fi
if [[ "${SUBSCRIBE_CAMERA}" == true ]]; then
  TOPICS+=(/avt_camera/left/image/compressed)
fi
rosbag play --clock --start "${START}" --duration "${DURATION}" \
  --rate "${RATE}" "${BAG}" --topics "${TOPICS[@]}" \
  >"${RESULT_DIR}/bag.log" 2>&1
sleep "${POST_RUN_WAIT}"
kill -INT "${STATUS_ECHO_PID}" 2>/dev/null || true
kill -INT "${STATS_ECHO_PID}" 2>/dev/null || true
wait "${STATUS_ECHO_PID}" 2>/dev/null || true
wait "${STATS_ECHO_PID}" 2>/dev/null || true
STATUS_ECHO_PID=""
STATS_ECHO_PID=""
timeout 10 rostopic echo -n 1 /hybrid/semantic_graph/stats \
  >"${RESULT_DIR}/graph_stats.txt" 2>&1 || true
timeout 10 rostopic echo -n 1 /hybrid/status \
  >"${RESULT_DIR}/frontend_status.txt" 2>&1 || true

kill -INT "${LAUNCH_PID}" 2>/dev/null || true
wait "${LAUNCH_PID}" 2>/dev/null || true
LAUNCH_PID=""
echo "Completed ${MODE}: ${RESULT_DIR}"
