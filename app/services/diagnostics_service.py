"""설치 환경 진단 (Phase 0). 경로가 잘못돼도 앱은 중단되지 않고 결과만 표시한다."""

from __future__ import annotations

import shutil
from pathlib import Path

from pydantic import BaseModel

from app.adapters.ports import CapabilityReport, EngineHealth, LyricsEnginePort, MusicEnginePort
from app.core.config import Settings
from app.domain.enums import Engine
from app.services.category_service import validate_adapter_dir


class DiagnosticItem(BaseModel):
    group: str
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""
    severity: str = "error"  # error | warning | info


class DiagnosticsReport(BaseModel):
    items: list[DiagnosticItem]
    ace: EngineHealth
    qwen: EngineHealth
    capabilities: CapabilityReport

    @property
    def ok_count(self) -> int:
        return sum(1 for i in self.items if i.ok)


class DiagnosticsService:
    def __init__(self, settings: Settings, music: MusicEnginePort, lyrics: LyricsEnginePort) -> None:
        self.s = settings
        self.music = music
        self.lyrics = lyrics

    def _path(self, group: str, name: str, path: Path, hint: str = "", is_dir: bool = True,
              severity: str = "error") -> DiagnosticItem:
        ok = path.is_dir() if is_dir else path.is_file()
        return DiagnosticItem(group=group, name=name, ok=ok, detail=str(path), hint="" if ok else hint,
                              severity=severity)

    def check_paths(self) -> list[DiagnosticItem]:
        s = self.s
        items = [
            self._path("경로", "ACE-Step 설치", s.ace_root, "STUDIO_ACE_ROOT 를 확인하세요."),
            self._path("경로", "ACE .venv Python", s.ace_python, "ACE 전용 가상환경이 필요합니다.", is_dir=False),
            self._path("경로", "ACE checkpoints", s.ace_root / "checkpoints", "모델 체크포인트 폴더가 없습니다.",
                       severity="warning"),
            self._path("경로", "lyrics-ai 설치", s.lyrics_root, "STUDIO_LYRICS_ROOT 를 확인하세요."),
            self._path("경로", "lyrics-ai .venv Python", s.lyrics_python, "MLX-LM 가상환경이 필요합니다.",
                       is_dir=False),
            self._path("보존 자산", "기존 데이터셋 test01", s.ace_root / "datasets" / "test01",
                       "없어도 앱은 동작합니다 (읽기 전용 보호 대상).", severity="info"),
        ]
        final_lora = s.ace_root / "lora_output" / "final_lora" / "adapter"
        ok, msg = validate_adapter_dir(Engine.ACE, final_lora)
        items.append(DiagnosticItem(group="보존 자산", name="기존 final_lora 어댑터", ok=ok,
                                    detail=f"{final_lora} — {msg}", severity="info"))
        dylib_ok = s.ffmpeg8_lib.is_dir() and any(s.ffmpeg8_lib.glob("libavcodec*"))
        items.append(DiagnosticItem(
            group="FFmpeg", name="FFmpeg 8 라이브러리 (TorchCodec 전처리용)", ok=dylib_ok,
            detail=str(s.ffmpeg8_lib),
            hint="" if dylib_ok else "brew install ffmpeg@8 후 ACE 프로세스에 DYLD_LIBRARY_PATH 전달 필요",
            severity="warning",
        ))
        system_ffmpeg = shutil.which("ffmpeg")
        items.append(DiagnosticItem(group="FFmpeg", name="시스템 ffmpeg", ok=bool(system_ffmpeg),
                                    detail=system_ffmpeg or "PATH 에 없음", severity="info"))
        for name, p in (("작업 폴더", s.workspace_dir), ("데이터 폴더", s.data_dir)):
            items.append(self._path("studio", name, p, "앱이 자동 생성합니다.", severity="warning"))
        return items

    async def run(self) -> DiagnosticsReport:
        items = self.check_paths()
        ace = await self.music.health()
        qwen = await self.lyrics.health()
        caps = await self.music.capabilities()
        items.append(DiagnosticItem(group="엔진", name="ACE-Step REST", ok=ace.ok, detail=ace.message,
                                    hint="" if ace.ok else "ACE 서버를 start_api_server_macos.sh 로 실행하세요."))
        items.append(DiagnosticItem(group="엔진", name="Qwen / MLX-LM", ok=qwen.ok,
                                    detail=f"{qwen.message} {qwen.version}".strip()))
        if ace.is_mock or qwen.is_mock:
            items.append(DiagnosticItem(group="엔진", name="Mock 모드", ok=False, severity="warning",
                                        detail="Mock 엔진 사용 중 — 결과는 실제 AI 생성물이 아닙니다."))
        return DiagnosticsReport(items=items, ace=ace, qwen=qwen, capabilities=caps)
