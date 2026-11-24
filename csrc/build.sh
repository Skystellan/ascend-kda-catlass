#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_EXECUTABLE:-$(which python3)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ascend_kda] ERROR: ASCEND_HOME_PATH is not set. Source CANN set_env.sh first." >&2
    exit 1
fi

CMAKE_ARGS=(
    -S "$ROOT"
    -B "$BUILD_DIR"
    "-DPython3_EXECUTABLE=$PY"
    -DCMAKE_BUILD_TYPE=Release
)
if [[ -n "${CATLASS_ROOT:-}" ]]; then
    CMAKE_ARGS+=("-DCATLASS_ROOT=$CATLASS_ROOT")
fi

mkdir -p "$BUILD_DIR"
cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" -j "$(nproc 2>/dev/null || echo 8)"
cp -f "$BUILD_DIR/libascend_kda_ops.so" "$ROOT/libascend_kda_ops.so"
echo "[ascend_kda] Built $ROOT/libascend_kda_ops.so"
