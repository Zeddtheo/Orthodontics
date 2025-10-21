#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MICROMAMBA_BIN="${MICROMAMBA_EXE:-micromamba}"

if ! command -v "${MICROMAMBA_BIN}" >/dev/null 2>&1; then
  if [[ -x "${SCRIPT_DIR}/build-tools/micromamba-linux" ]]; then
    MICROMAMBA_BIN="${SCRIPT_DIR}/build-tools/micromamba-linux"
  else
    echo "[ERROR] micromamba not found. Set MICROMAMBA_EXE or place micromamba in PATH." >&2
    exit 1
  fi
fi

if [[ $# -gt 0 ]]; then
  ARGS=("$@")
else
  UPPER_STL=${UPPER_STL:-}
  LOWER_STL=${LOWER_STL:-}
  OUTPUT_JSON=${OUTPUT_JSON:-}
  KEEP_INTERMEDIATE=${KEEP_INTERMEDIATE:-0}
  PRETTY_JSON=${PRETTY_JSON:-0}

  if [[ -z "${UPPER_STL}" || -z "${LOWER_STL}" ]]; then
    cat <<'EOF' >&2
[ERROR] 请提供输入 STL。用法：
  ./start.sh --upper /path/to/U.stl --lower /path/to/L.stl [--output out.json] [--keep] [--pretty]
或通过环境变量：
  UPPER_STL=/path/to/U.stl LOWER_STL=/path/to/L.stl OUTPUT_JSON=/path/result.json ./start.sh
EOF
    exit 1
  fi

  ARGS=(--upper "${UPPER_STL}" --lower "${LOWER_STL}")
  if [[ -n "${OUTPUT_JSON}" ]]; then
    ARGS+=(--output "${OUTPUT_JSON}")
  fi
  if [[ "${KEEP_INTERMEDIATE}" == "1" || "${KEEP_INTERMEDIATE,,}" == "true" ]]; then
    ARGS+=(--keep)
  fi
  if [[ "${PRETTY_JSON}" == "1" || "${PRETTY_JSON,,}" == "true" ]]; then
    ARGS+=(--pretty)
  fi
fi

exec "${MICROMAMBA_BIN}" run -n calc python "${SCRIPT_DIR}/run_pipeline.py" "${ARGS[@]}"
