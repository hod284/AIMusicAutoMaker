"""앱 설정 (싱글톤).

- 우선순위: 환경변수 > ``studio/.env`` > 기본값
- 외부 라이브러리 없이 ``.env`` 를 파싱한다 (studio/.venv 의존성 최소화).
- 경로는 모두 ``Path`` 로 정규화한다. 기존 ACE / lyrics-ai 설치본은 **읽기 전용 참조**로만 쓴다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.core.singleton import SingletonMeta

STUDIO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _expand(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


@dataclass
class Settings:
    """설정 값 객체. ``AppSettings().get()`` 로 싱글톤 인스턴스를 얻는다."""

    # --- 경로 ---
    music_root: Path = field(default_factory=lambda: _expand("~/AI-Music-Studio"))
    studio_root: Path = STUDIO_ROOT
    ace_root: Path = field(default_factory=lambda: _expand("~/AI-Music-Studio/ACE-Step-1.5"))
    lyrics_root: Path = field(default_factory=lambda: _expand("~/AI-Music-Studio/lyrics-ai"))
    workspace_dir: Path = STUDIO_ROOT / "workspace"
    data_dir: Path = STUDIO_ROOT / "data"
    ffmpeg8_lib: Path = Path("/opt/homebrew/opt/ffmpeg@8/lib")
    ffmpeg8_bin: Path = Path("/opt/homebrew/opt/ffmpeg@8/bin")

    # --- 서버 ---
    ui_host: str = "127.0.0.1"
    ui_port: int = 8080
    ace_base_url: str = "http://127.0.0.1:8001"
    ace_api_key: str = ""  # 비어 있으면 인증 미사용. DB/로그에 절대 기록하지 않는다.
    ace_route_bindings: Path = STUDIO_ROOT / "config" / "ace_routes.json"

    # --- 모델 ---
    qwen_base_model: str = "mlx-community/Qwen3-4B-4bit"
    ace_default_model: str = ""  # 비어 있으면 서버 기본 모델

    # --- 엔진 선택 (어댑터 팩토리) ---
    music_engine: str = "ace"   # ace | mock
    lyrics_engine: str = "mlx"  # mlx | mock

    # --- 정책 ---
    max_upload_mb: int = 200
    gpu_heavy_concurrency: int = 1
    ace_poll_interval_s: float = 2.0
    ace_poll_timeout_s: float = 1800.0
    http_timeout_s: float = 30.0
    lyrics_timeout_s: float = 600.0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "studio.db"

    @property
    def ace_python(self) -> Path:
        return self.ace_root / ".venv" / "bin" / "python"

    @property
    def lyrics_python(self) -> Path:
        return self.lyrics_root / ".venv" / "bin" / "python"

    @property
    def categories_dir(self) -> Path:
        return self.workspace_dir / "categories"

    @property
    def ace_studio_share_dir(self) -> Path:
        """ACE 의 path allowlist 대응용 공유 폴더 (기존 datasets/test01 과 분리)."""
        return self.ace_root / "datasets" / "studio"

    @property
    def protected_paths(self) -> list[Path]:
        """절대 쓰기/삭제하면 안 되는 기존 자산."""
        return [
            self.ace_root / "datasets" / "test01",
            self.ace_root / "lora_output",
            self.ace_root / "checkpoints",
            self.lyrics_root / ".venv",
            self.ace_root / ".venv",
        ]

    def ace_subprocess_env(self) -> dict[str, str]:
        """ACE 하위 프로세스용 환경. FFmpeg 8 라이브러리 경로를 명시적으로 전달한다."""
        env = dict(os.environ)
        env["PATH"] = f"{self.ffmpeg8_bin}:{env.get('PATH', '')}"
        env["DYLD_LIBRARY_PATH"] = str(self.ffmpeg8_lib)
        return env


_ENV_MAP: dict[str, tuple[str, type]] = {
    "STUDIO_MUSIC_ROOT": ("music_root", Path),
    "STUDIO_ACE_ROOT": ("ace_root", Path),
    "STUDIO_LYRICS_ROOT": ("lyrics_root", Path),
    "STUDIO_WORKSPACE_DIR": ("workspace_dir", Path),
    "STUDIO_DATA_DIR": ("data_dir", Path),
    "STUDIO_FFMPEG8_LIB": ("ffmpeg8_lib", Path),
    "STUDIO_FFMPEG8_BIN": ("ffmpeg8_bin", Path),
    "STUDIO_UI_HOST": ("ui_host", str),
    "STUDIO_UI_PORT": ("ui_port", int),
    "ACE_BASE_URL": ("ace_base_url", str),
    "ACESTEP_API_KEY": ("ace_api_key", str),
    "ACE_ROUTE_BINDINGS": ("ace_route_bindings", Path),
    "QWEN_BASE_MODEL": ("qwen_base_model", str),
    "ACE_DEFAULT_MODEL": ("ace_default_model", str),
    "STUDIO_MUSIC_ENGINE": ("music_engine", str),
    "STUDIO_LYRICS_ENGINE": ("lyrics_engine", str),
    "STUDIO_MAX_UPLOAD_MB": ("max_upload_mb", int),
    "STUDIO_GPU_CONCURRENCY": ("gpu_heavy_concurrency", int),
    "ACE_POLL_INTERVAL_S": ("ace_poll_interval_s", float),
    "ACE_POLL_TIMEOUT_S": ("ace_poll_timeout_s", float),
    "STUDIO_LYRICS_TIMEOUT_S": ("lyrics_timeout_s", float),
}


class AppSettings(metaclass=SingletonMeta):
    """설정 싱글톤. 앱 전체가 같은 ``Settings`` 인스턴스를 공유한다."""

    def __init__(self, overrides: dict[str, object] | None = None, env_file: Path | None = None) -> None:
        self._settings = self._build(overrides or {}, env_file or STUDIO_ROOT / ".env")

    def get(self) -> Settings:
        return self._settings

    @staticmethod
    def _build(overrides: dict[str, object], env_file: Path) -> Settings:
        merged: dict[str, str] = {**_load_dotenv(env_file), **os.environ}
        kwargs: dict[str, object] = {}
        for env_key, (attr, typ) in _ENV_MAP.items():
            if env_key in merged and merged[env_key] != "":
                raw = merged[env_key]
                kwargs[attr] = _expand(raw) if typ is Path else typ(raw)
        kwargs.update(overrides)
        settings = Settings(**kwargs)  # type: ignore[arg-type]
        # music_root 만 바꾼 경우 하위 경로를 따라가게 한다
        if "music_root" in kwargs:
            if "ace_root" not in kwargs:
                settings.ace_root = settings.music_root / "ACE-Step-1.5"
            if "lyrics_root" not in kwargs:
                settings.lyrics_root = settings.music_root / "lyrics-ai"
        return settings


def get_settings() -> Settings:
    return AppSettings().get()
