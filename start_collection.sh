#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$(readlink -f -- "$0")")"
export MUJOCO_GL="${MUJOCO_GL:-glx}"
exec "${DPPO_PYTHON:-python}" -B collect_phone_demos.py "$@"
