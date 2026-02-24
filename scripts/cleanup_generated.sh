#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

remove_if_exists() {
    local path="$1"
    if [ -e "$path" ]; then
        rm -rf "$path"
        echo "Removido: $path"
    else
        echo "Ignorado (nao existe): $path"
    fi
}

remove_if_exists "$ROOT_DIR/versions/v3/BATCH_OUTPUT"
remove_if_exists "$ROOT_DIR/versions/v3/CNN/outputs"
remove_if_exists "$ROOT_DIR/versions/v3/geracnn/yolo"

find "$ROOT_DIR/versions" -type d -name '__pycache__' -prune -exec rm -rf {} +
echo "Limpeza concluida."
