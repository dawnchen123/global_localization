#!/usr/bin/env bash

export PHDEXP_EVO_ROOT="/home/dawn/document/phd_exp"
export PHDEXP_EVO_HOME="${PHDEXP_EVO_ROOT}/evo_home"
mkdir -p "${PHDEXP_EVO_HOME}"

evo_ape() {
  HOME="${PHDEXP_EVO_HOME}" MPLBACKEND=Agg "${PHDEXP_EVO_ROOT}/evo_env/bin/evo_ape" "$@"
}

evo_rpe() {
  HOME="${PHDEXP_EVO_HOME}" MPLBACKEND=Agg "${PHDEXP_EVO_ROOT}/evo_env/bin/evo_rpe" "$@"
}

evo_traj() {
  HOME="${PHDEXP_EVO_HOME}" MPLBACKEND=Agg "${PHDEXP_EVO_ROOT}/evo_env/bin/evo_traj" "$@"
}

evo_res() {
  HOME="${PHDEXP_EVO_HOME}" MPLBACKEND=Agg "${PHDEXP_EVO_ROOT}/evo_env/bin/evo_res" "$@"
}

evo_config() {
  HOME="${PHDEXP_EVO_HOME}" MPLBACKEND=Agg "${PHDEXP_EVO_ROOT}/evo_env/bin/evo_config" "$@"
}

echo "Loaded phd_exp evo helpers. Try: evo_ape tum --help"
