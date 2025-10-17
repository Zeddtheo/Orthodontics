#!/usr/bin/env bash
set -euo pipefail

API_URL=${1:-http://localhost:8000/infer}

curl -s -X POST "${API_URL}" \
  -H "Content-Type: application/json" \
  -d "{\"path_a\": \"/app/tests/A.stl\", \"path_b\": \"/app/tests/B.stl\", \"include_intermediate\": false}" \
  | python -m json.tool
