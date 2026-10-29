#!/bin/bash
# Studio 와 음악 엔진(ACE)을 끕니다 (진행 중인 작업이 있으면 끄지 않음).
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
LOG_DIR="$STUDIO_DIR/data/launcher"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/launcher.log"
echo "===== $(date '+%Y-%m-%d %H:%M:%S') --stop =====" >> "$LOG"
"$STUDIO_DIR/.venv/bin/python" "$STUDIO_DIR/scripts/launch_studio.py" --stop "$@" 2>&1 | tee -a "$LOG"
result=${PIPESTATUS[0]}
echo "(종료 코드 $result)" >> "$LOG"
echo
echo "실행 기록: $LOG"
echo "음악 엔진(ACE) 출력: $LOG_DIR/ace.log"
if [[ $result -ne 0 ]]; then
    read -r -p "Enter를 누르면 종료합니다. "
fi
exit "$result"
