"""Mock 음악 엔진. 실제 ACE 없이 UI/파이프라인을 개발·테스트하기 위한 어댑터.

출력은 짧은 사인파 WAV 이며, 결과 메타데이터에 ``is_mock: true`` 를 남겨
실제 생성 결과로 오인하지 않게 한다.
"""

from __future__ import annotations

import asyncio
import math
import struct
import tempfile
import uuid
import wave
from pathlib import Path

from app.adapters.ports import (
    Capability,
    CapabilityReport,
    CapabilityState,
    EngineHealth,
    GeneratedAudio,
    LoraStatus,
    MusicEnginePort,
    MusicGenerationRequest,
    TaskResult,
    TaskState,
)
from app.adapters.ace.capabilities import CAPABILITY_LABELS
from app.utils.fileio import atomic_write_bytes


def _sine_wav(path: Path, seconds: float = 2.0, freq: float = 440.0, rate: int = 22050) -> None:
    frames = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate))) for i in range(int(seconds * rate))
    )
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames)


class MockMusicAdapter(MusicEnginePort):
    name = "ace-mock"
    is_mock = True

    def __init__(self, latency_s: float = 0.05, polls_until_done: int = 1, fail: bool = False) -> None:
        self.latency_s = latency_s
        self.polls_until_done = polls_until_done
        self.fail = fail
        self._tasks: dict[str, dict] = {}
        self._lora: str | None = None
        self._tmp = Path(tempfile.mkdtemp(prefix="ace-mock-"))
        self.submitted: list[MusicGenerationRequest] = []

    async def health(self) -> EngineHealth:
        return EngineHealth(engine="ace", ok=True, message="Mock 엔진 (실제 ACE 아님)", is_mock=True)

    async def capabilities(self, refresh: bool = False) -> CapabilityReport:
        caps = {
            n: Capability(name=n, label_ko=label, state=CapabilityState.MOCK, reason="Mock")
            for n, label in CAPABILITY_LABELS.items()
        }
        return CapabilityReport(engine="ace", capabilities=caps, source="mock")

    async def submit_generation(self, req: MusicGenerationRequest) -> str:
        await asyncio.sleep(self.latency_s)
        task_id = f"mock-{uuid.uuid4().hex[:8]}"
        self.submitted.append(req)
        self._tasks[task_id] = {"polls": 0, "req": req, "lora": self._lora}
        return task_id

    async def query_results(self, task_ids: list[str]) -> list[TaskResult]:
        results = []
        for tid in task_ids:
            task = self._tasks.get(tid)
            if task is None:
                results.append(TaskResult(task_id=tid, state=TaskState.FAILED, error="unknown task"))
                continue
            task["polls"] += 1
            if task["polls"] < self.polls_until_done:
                results.append(TaskResult(task_id=tid, state=TaskState.RUNNING))
            elif self.fail:
                results.append(TaskResult(task_id=tid, state=TaskState.FAILED, error="mock failure"))
            else:
                wav = self._tmp / f"{tid}.wav"
                if not wav.exists():
                    _sine_wav(wav, freq=523.25 if task["lora"] else 440.0)
                seed = task["req"].seed if task["req"].seed is not None else 1234
                results.append(TaskResult(task_id=tid, state=TaskState.SUCCEEDED, audios=[
                    GeneratedAudio(file=str(wav), seed=str(seed), dit_model="mock-dit",
                                   metas={"lora": task["lora"], "is_mock": True})
                ]))
        return results

    async def download_audio(self, audio: GeneratedAudio, dest: Path) -> Path:
        src = Path(audio.file)
        return atomic_write_bytes(dest.with_suffix(src.suffix), src.read_bytes())

    async def load_lora(self, adapter_path: Path) -> LoraStatus:
        self._lora = str(adapter_path)
        return await self.lora_status()

    async def unload_lora(self) -> LoraStatus:
        self._lora = None
        return await self.lora_status()

    async def lora_status(self) -> LoraStatus:
        return LoraStatus(loaded=self._lora is not None, enabled=self._lora is not None,
                          adapter_path=self._lora, verified_by="mock")
