"""테스트·데모용 가짜 목소리 엔진. 실제 AI 를 돌리지 않고 짧은 wav 를 만든다 (결과에 MOCK 표시)."""

from __future__ import annotations

import asyncio
import math
import struct
import wave
from pathlib import Path

from app.adapters.ports import (
    EngineHealth,
    LogFn,
    ProgressCallback,
    TrainHandle,
    TrainProgress,
    VoiceConvertSpec,
    VoiceEnginePort,
    VoiceTrainSpec,
)
from app.core.errors import EngineResponseError, ValidationError


def write_tone(path: Path, seconds: float = 2.0, freq: float = 440.0, sr: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / sr)))
                          for i in range(int(seconds * sr)))
        w.writeframes(frames)
    return path


class _MockHandle(TrainHandle):
    def __init__(self, task: asyncio.Task) -> None:
        self.task = task

    async def wait(self) -> int:
        return await self.task

    async def cancel(self, grace_s: float = 10.0) -> None:
        self.task.cancel()


class MockVoiceAdapter(VoiceEnginePort):
    name = "mock"
    is_mock = True

    def __init__(self, latency_s: float = 0.0, train_steps_delay_s: float = 0.0) -> None:
        self.latency_s = latency_s
        self.train_steps_delay_s = train_steps_delay_s
        self.fail_convert = False
        self.calls: list[str] = []
        self.last_convert: VoiceConvertSpec | None = None

    async def health(self) -> EngineHealth:
        return EngineHealth(engine="voice", ok=True, message="Mock 목소리 엔진", is_mock=True)

    async def separate(self, src: Path, out_dir: Path, log: LogFn) -> tuple[Path, Path]:
        self.calls.append("separate")
        await asyncio.sleep(self.latency_s)
        if not Path(src).is_file():
            raise ValidationError("원본 파일이 없습니다.", detail=str(src))
        log("[mock] 보컬/반주 분리")
        return write_tone(Path(out_dir) / "vocals.wav", 3.0, 330), write_tone(Path(out_dir) / "instrumental.wav", 3.0, 110)

    async def convert(self, spec: VoiceConvertSpec, log: LogFn) -> Path:
        self.calls.append("convert")
        self.last_convert = spec
        await asyncio.sleep(self.latency_s)
        if self.fail_convert:
            raise EngineResponseError("[mock] 변환 실패")
        log("[mock] 목소리 변환")
        return write_tone(Path(spec.output_dir) / "vc_vocals_reference_1.0_30_0.7.wav", 3.0, 520)

    async def start_training(self, spec: VoiceTrainSpec, on_progress: ProgressCallback) -> TrainHandle:
        self.calls.append("train")

        async def _run() -> int:
            out = spec.log_dir / spec.run_name
            for step in range(0, spec.max_steps, max(1, spec.max_steps // 5)):
                await asyncio.sleep(self.train_steps_delay_s)
                on_progress(TrainProgress(step=step, total=spec.max_steps, train_loss=1.0 / (step + 1),
                                          line=f"epoch 0, step {step}, loss: {1.0 / (step + 1)}"))
            out.mkdir(parents=True, exist_ok=True)
            (out / "ft_model.pth").write_bytes(b"mock")
            (out / spec.preset).write_text("log_dir: mock\n")
            (out / "DiT_epoch_00000_step_00010.pth").write_bytes(b"mock-intermediate")
            on_progress(TrainProgress(line=f"Final model saved at {out / 'ft_model.pth'}"))
            return 0

        return _MockHandle(asyncio.create_task(_run()))

    async def split_clips(self, src: Path, out_dir: Path, seconds: float = 15.0) -> list[Path]:
        self.calls.append("split")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        return [write_tone(out / f"{Path(src).stem}_{i:03d}.wav", 2.0 + i, 300 + 50 * i) for i in range(2)]

    async def mix(self, vocals: Path, instrumental: Path, dest: Path) -> Path:
        self.calls.append("mix")
        if Path(dest).exists():
            raise ValidationError("같은 이름의 파일이 이미 있습니다 (덮어쓰지 않음).", detail=str(dest))
        return write_tone(Path(dest), 3.0, 440)
