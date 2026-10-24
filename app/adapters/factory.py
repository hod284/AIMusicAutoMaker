"""어댑터 팩토리 (Factory + Registry).

설정의 ``music_engine`` / ``lyrics_engine`` 문자열로 구현체를 선택한다.
새 엔진은 ``AdapterFactory.register_music("이름", builder)`` 한 줄로 추가 — 기존 코드 수정 불필요(OCP).
"""

from __future__ import annotations

from typing import Callable

from app.adapters.ace.capabilities import AceCapabilityRegistry
from app.adapters.ace.mock_adapter import MockMusicAdapter
from app.adapters.ace.rest_adapter import AceRestAdapter
from app.adapters.mlx.mock_adapter import MockLyricsAdapter
from app.adapters.mlx.process_adapter import MlxProcessAdapter
from app.adapters.ports import LyricsEnginePort, MusicEnginePort
from app.core.config import Settings
from app.core.errors import ValidationError

MusicBuilder = Callable[[Settings], MusicEnginePort]
LyricsBuilder = Callable[[Settings], LyricsEnginePort]


class AdapterFactory:
    _music: dict[str, MusicBuilder] = {}
    _lyrics: dict[str, LyricsBuilder] = {}

    @classmethod
    def register_music(cls, name: str, builder: MusicBuilder) -> None:
        cls._music[name] = builder

    @classmethod
    def register_lyrics(cls, name: str, builder: LyricsBuilder) -> None:
        cls._lyrics[name] = builder

    @classmethod
    def music(cls, settings: Settings) -> MusicEnginePort:
        try:
            return cls._music[settings.music_engine](settings)
        except KeyError as exc:
            raise ValidationError(f"알 수 없는 음악 엔진: {settings.music_engine}") from exc

    @classmethod
    def lyrics(cls, settings: Settings) -> LyricsEnginePort:
        try:
            return cls._lyrics[settings.lyrics_engine](settings)
        except KeyError as exc:
            raise ValidationError(f"알 수 없는 가사 엔진: {settings.lyrics_engine}") from exc

    @classmethod
    def available(cls) -> dict[str, list[str]]:
        return {"music": sorted(cls._music), "lyrics": sorted(cls._lyrics)}


AdapterFactory.register_music(
    "ace",
    lambda s: AceRestAdapter(
        s.ace_base_url, AceCapabilityRegistry(s.ace_route_bindings), api_key=s.ace_api_key,
        timeout_s=s.http_timeout_s,
    ),
)
AdapterFactory.register_music("mock", lambda s: MockMusicAdapter())
AdapterFactory.register_lyrics(
    "mlx", lambda s: MlxProcessAdapter(s.lyrics_python, s.qwen_base_model, s.lyrics_root, s.lyrics_timeout_s)
)
AdapterFactory.register_lyrics("mock", lambda s: MockLyricsAdapter())
