#!/bin/bash
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDIO_DIR="$SCRIPT_DIR"
if [[ ! -f "$STUDIO_DIR/scripts/launch_studio.py" ]]; then
    STUDIO_DIR="/Users/jeong-yunsu/AI-Music-Studio/studio"
fi
if [[ ! -x "$STUDIO_DIR/.venv/bin/python" || ! -f "$STUDIO_DIR/scripts/launch_studio.py" ]]; then
    echo "Studio 실행 파일을 찾을 수 없습니다: $STUDIO_DIR"
    read -r -p "Enter를 누르면 종료합니다. "
    exit 1
fi
cd "$STUDIO_DIR" || exit 1
"$STUDIO_DIR/.venv/bin/python" "$STUDIO_DIR/scripts/launch_studio.py" --restart "$@"
result=$?
if [[ $result -ne 0 ]]; then
    read -r -p "Enter를 누르면 종료합니다. "
fi
exit "$result"
