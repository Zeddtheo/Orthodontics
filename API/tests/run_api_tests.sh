#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
API_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

ROOT_PREFIX_DEFAULT="${API_ROOT}/.micromamba"
ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${ROOT_PREFIX_DEFAULT}}"
export MAMBA_ROOT_PREFIX="${ROOT_PREFIX}"

MICROMAMBA_BIN="${MICROMAMBA_EXE:-}"
if [[ -z "${MICROMAMBA_BIN}" ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    MICROMAMBA_BIN="$(command -v micromamba)"
  elif [[ -x "${API_ROOT}/build-tools/micromamba-linux" ]]; then
    MICROMAMBA_BIN="${API_ROOT}/build-tools/micromamba-linux"
  else
    echo "[ERROR] 未找到 micromamba，可通过安装后设置 MICROMAMBA_EXE 环境变量指向其可执行文件。" >&2
    exit 1
  fi
fi
export MICROMAMBA_EXE="${MICROMAMBA_BIN}"

ensure_env() {
  local env_name="$1"
  local spec_file="$2"
  local env_dir="${ROOT_PREFIX}/envs/${env_name}"
  if [[ -d "${env_dir}" ]]; then
    return 0
  fi
  echo "[INFO] 创建 micromamba 环境 ${env_name}..."
  "${MICROMAMBA_BIN}" create -y -n "${env_name}" -f "${API_ROOT}/envs/${spec_file}" --root-prefix "${ROOT_PREFIX}"
}

ensure_env "calc" "calc.yml"
ensure_env "ios_model" "ios_model.yml"
ensure_env "pointnetreg" "pointnetreg.yml"

UPPER_STL="${1:-${SCRIPT_DIR}/319_U.stl}"
if [[ $# -ge 1 ]]; then
  shift
fi
LOWER_STL="${1:-${SCRIPT_DIR}/319_L.stl}"
if [[ $# -ge 1 ]]; then
  shift
fi

WORKDIR="$(mktemp -d "${API_ROOT}/tests/tmp_api_pipeline.XXXXXX")"
cleanup() {
  rm -rf "${WORKDIR}"
}
trap cleanup EXIT

OUTPUT_JSON="${WORKDIR}/result.json"

echo "[INFO] 运行 ios-model + PointNet-Reg 集成管线..."
"${API_ROOT}/run.sh" \
  --upper "${UPPER_STL}" \
  --lower "${LOWER_STL}" \
  --output "${OUTPUT_JSON}" \
  "$@"

echo "[INFO] 验证输出 JSON 结构..."
"${MICROMAMBA_BIN}" run --root-prefix "${ROOT_PREFIX}" -n calc python "${SCRIPT_DIR}/verify_pipeline_output.py" --json "${OUTPUT_JSON}"

echo "[INFO] API 集成测试通过。"
