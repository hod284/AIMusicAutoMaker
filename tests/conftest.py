from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.adapters.ace.mock_adapter import MockMusicAdapter  # noqa: E402
from app.adapters.mlx.mock_adapter import MockLyricsAdapter  # noqa: E402
from app.adapters.voice.mock_adapter import MockVoiceAdapter  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.singleton import SingletonMeta  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_singletons():
    SingletonMeta.reset_all()
    yield
    SingletonMeta.reset_all()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    ace = tmp_path / "ACE-Step-1.5"
    # 보호 대상 기존 자산 흉내
    (ace / "datasets" / "test01").mkdir(parents=True)
    (ace / "datasets" / "test01" / "my_lora_dataset.json").write_text("{}")
    adapter = ace / "lora_output" / "final_lora" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r": 8}')
    (adapter / "adapter_model.safetensors").write_bytes(b"\x00" * 16)
    return Settings(
        music_root=tmp_path, ace_root=ace, lyrics_root=tmp_path / "lyrics-ai",
        workspace_dir=tmp_path / "studio" / "workspace", data_dir=tmp_path / "studio" / "data",
        ace_poll_interval_s=0.01, ace_poll_timeout_s=5, music_engine="mock", lyrics_engine="mock",
        voice_engine="mock", voice_root=tmp_path / "voice-ai",
    )


@pytest.fixture
def music() -> MockMusicAdapter:
    return MockMusicAdapter(latency_s=0, polls_until_done=2)


@pytest.fixture
def lyrics() -> MockLyricsAdapter:
    return MockLyricsAdapter(latency_s=0)


@pytest.fixture
def voice() -> MockVoiceAdapter:
    return MockVoiceAdapter()


@pytest.fixture
def container(settings, music, lyrics, voice):
    from app.core.container import ServiceContainer

    return ServiceContainer(settings, music=music, lyrics=lyrics, voice=voice)


@pytest.fixture
def ballad(container):
    return container.categories.create(
        "2000년대 감성 록 발라드",
        style={"allowed_genres": ["rock ballad"], "vocal": "male", "preferred_bpm_min": 70,
               "preferred_bpm_max": 85, "instruments": ["piano", "electric guitar"]},
    )

pytest_plugins = ["nicegui.testing.user_plugin"]
