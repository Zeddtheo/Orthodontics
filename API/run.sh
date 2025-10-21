#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR="${SCRIPT_DIR}"

MICROMAMBA_BIN="${MICROMAMBA_EXE:-}"
if [[ -z "${MICROMAMBA_BIN}" ]]; then
  if command -v micromamba >/dev/null 2>&1; then
    MICROMAMBA_BIN="$(command -v micromamba)"
  elif [[ -x "${ROOT_DIR}/build-tools/micromamba-linux" ]]; then
    MICROMAMBA_BIN="${ROOT_DIR}/build-tools/micromamba-linux"
  else
    echo "[ERROR] 未找到 micromamba，请先安装或设置 MICROMAMBA_EXE。" >&2
    exit 1
  fi
fi

if [[ $# -eq 0 ]]; then
  cat <<'EOF'
用法: ./run.sh --upper PATH --lower PATH [--output PATH] [--keep] [--pretty]
该脚本会调用 micromamba calc 环境里的 run_pipeline.py。
EOF
  exit 1
fi

exec "${MICROMAMBA_BIN}" run -n calc python "${ROOT_DIR}/run_pipeline.py" "$@"
