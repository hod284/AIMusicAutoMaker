"""AI Music Studio 진입점.

실행:  cd ~/AI-Music-Studio/studio && .venv/bin/python main.py
Mock:  STUDIO_MUSIC_ENGINE=mock STUDIO_LYRICS_ENGINE=mock .venv/bin/python main.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from nicegui import app, ui

from app.core.config import get_settings
from app.core.container import ServiceContainer
from app.core.logging_setup import setup_logging

# 페이지 등록 (import 시 @ui.page 데코레이터가 라우트를 만든다)
from app.views import (  # noqa: F401
    create_song_view,
    dashboard_view,
    lyrics_book_view,
    music_training_view,
    other_views,
)

settings = get_settings()
setup_logging(settings.data_dir / "logs")
log = logging.getLogger("studio")
container = ServiceContainer(settings)

MEDIA_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".flac": "audio/flac", ".opus": "audio/ogg",
               ".ogg": "audio/ogg", ".aac": "audio/aac", ".m4a": "audio/mp4"}


@app.get("/media/songs/{song_id}")
def serve_song(song_id: str, download: int = 0) -> FileResponse:
    """DB 에 등록된 곡만, workspace 내부 경로일 때만 제공 (임의 파일 열람 차단)."""
    song = container.song_repo.get(song_id)
    if song is None or not song.audio_path:
        raise HTTPException(404, "곡을 찾을 수 없습니다.")
    # 경로는 사용자가 아니라 DB(앱이 저장한 기록)에서만 온다 → 사용자 지정 결과 폴더도 재생 허용
    path = Path(song.audio_path).resolve()
    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None or not path.is_file():
        raise HTTPException(404, "오디오 파일이 없습니다.")
    filename = f"{(song.title or 'song').replace('/', '_')}{path.suffix}"
    return FileResponse(path, media_type=media_type, filename=filename if download else None)


app.on_startup(container.startup)
app.on_shutdown(container.shutdown)

if __name__ in {"__main__", "__mp_main__"}:
    log.info("AI Music Studio 시작: http://%s:%s (음악=%s, 가사=%s)", settings.ui_host, settings.ui_port,
             settings.music_engine, settings.lyrics_engine)
    ui.run(
        host=settings.ui_host,          # 기본 127.0.0.1 — 외부 접속 불허
        port=settings.ui_port,
        title="AI Music Studio",
        language="ko-KR",
        reload=False,                   # 워커/잡 상태 보존을 위해 자동 리로드 끔
        show=False,
        favicon="🎵",
    )
