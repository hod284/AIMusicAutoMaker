#!/usr/bin/env bash
# Phase 0 설치본 확인 (읽기 전용 — 어떤 파일도 수정/삭제하지 않음)
# 사용법: cd ~/AI-Music-Studio/studio && bash scripts/verify_install.sh
set -u
ROOT="${STUDIO_MUSIC_ROOT:-$HOME/AI-Music-Studio}"
ACE="$ROOT/ACE-Step-1.5"
LYR="$ROOT/lyrics-ai"
ACE_URL="${ACE_BASE_URL:-http://127.0.0.1:8001}"
OUT="$(pwd)/data/install_report"
mkdir -p "$OUT"

echo "== 1. 경로 =="
for p in "$ACE" "$ACE/.venv/bin/python" "$ACE/datasets/test01" "$ACE/lora_output/final_lora/adapter/adapter_config.json" \
         "$ACE/lora_output/final_lora/adapter/adapter_model.safetensors" "$LYR/.venv/bin/python" \
         /opt/homebrew/opt/ffmpeg@8/lib; do
  [ -e "$p" ] && echo "  OK   $p" || echo "  없음 $p"
done

echo "== 2. ACE API openapi.json (서버가 실행 중이어야 함) =="
if curl -fsS "$ACE_URL/openapi.json" -o "$OUT/ace-openapi.json"; then
  python3 - "$OUT/ace-openapi.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
for path, methods in sorted(j.get("paths", {}).items()):
    print(" ", ",".join(m.upper() for m in methods), path)
PY
  echo "  → 저장: $OUT/ace-openapi.json"
else
  echo "  ACE 서버에 연결 실패 ($ACE_URL). start_api_server_macos.sh 로 실행 후 다시 시도"
fi

echo "== 3. ACE 라우터 소스 (LoRA/학습/데이터셋 관련 라우트 데코레이터) =="
grep -rnE '@(app|router)\.(get|post|put|delete)\(' "$ACE/acestep" 2>/dev/null \
  | grep -iE 'lora|train|dataset|preprocess|label|scan|export' | sed 's|^|  |' | tee "$OUT/ace-routes-grep.txt" | head -60

echo "== 4. start_api_server_macos.sh 의 FFmpeg/DYLD 전달 여부 =="
grep -nE 'DYLD|ffmpeg|FFMPEG' "$ACE/start_api_server_macos.sh" 2>/dev/null | sed 's|^|  |' || echo "  (해당 없음)"

echo "== 5. MLX-LM =="
"$LYR/.venv/bin/python" -c "import mlx_lm; print('  mlx_lm', getattr(mlx_lm,'__version__','?'))" 2>&1
"$LYR/.venv/bin/python" -m mlx_lm.lora --help > "$OUT/mlx_lora_help.txt" 2>&1 \
  && echo "  → mlx_lm.lora --help 저장: $OUT/mlx_lora_help.txt" || echo "  mlx_lm.lora --help 실패"
"$LYR/.venv/bin/python" -m mlx_lm.generate --help > "$OUT/mlx_generate_help.txt" 2>&1 \
  && echo "  → mlx_lm.generate --help 저장" || echo "  mlx_lm.generate --help 실패"

echo "== 6. Qwen 워커 실제 호출 (모델 로드 → 짧은 생성) =="
echo '{"model":"mlx-community/Qwen3-4B-4bit","user":"한 줄로 인사해줘","max_tokens":40,"temperature":0.7}' \
  | "$LYR/.venv/bin/python" app/workers/mlx_generate_worker.py 2>"$OUT/mlx_worker_stderr.txt" | tail -1
echo "완료. 결과 폴더: $OUT"
