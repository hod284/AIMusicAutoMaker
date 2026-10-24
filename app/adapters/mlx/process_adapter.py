"""MLX-LM 서브프로세스 어댑터 (어댑터 패턴: lyrics-ai/.venv 프로세스 → LyricsEnginePort).

- 추론: ``lyrics-ai/.venv/bin/python app/workers/mlx_generate_worker.py`` 에 JSON 을 stdin 으로 전달
- 학습: ``python -m mlx_lm.lora --help`` 로 옵션을 확인한 뒤 **지원되는 옵션만** 인자 리스트로 전달
- ``shell=True`` 를 절대 쓰지 않는다 (asyncio.create_subprocess_exec)
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import signal
import time
from pathlib import Path

from app.adapters.ports import (
    EngineHealth,
    LLMRequest,
    LLMResult,
    LoraTrainSpec,
    LyricsEnginePort,
    ProgressCallback,
    TrainHandle,
    TrainProgress,
)
from app.core.errors import CapabilityUnsupportedError, EngineResponseError, EngineUnavailableError

log = logging.getLogger(__name__)

WORKER_SCRIPT = Path(__file__).resolve().parents[2] / "workers" / "mlx_generate_worker.py"
RESULT_PREFIX = "@@RESULT@@"
_FLAG_RE = re.compile(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)")
_ITER_TRAIN_RE = re.compile(r"Iter\s+(\d+):\s*Train loss\s+([0-9.]+)", re.IGNORECASE)
_ITER_VAL_RE = re.compile(r"Iter\s+(\d+):\s*Val loss\s+([0-9.]+)", re.IGNORECASE)

# 앱 필드 → mlx_lm.lora CLI 플래그. 실제 사용 여부는 --help 결과로 필터링한다.
_TRAIN_FLAG_MAP = {
    "iters": "--iters",
    "batch_size": "--batch-size",
    "learning_rate": "--learning-rate",
    "max_seq_length": "--max-seq-length",
    "num_layers": "--num-layers",
    "save_every": "--save-every",
    "steps_per_eval": "--steps-per-eval",
}


def parse_help_flags(help_text: str) -> set[str]:
    return set(_FLAG_RE.findall(help_text))


def parse_train_line(line: str) -> TrainProgress | None:
    if m := _ITER_TRAIN_RE.search(line):
        return TrainProgress(step=int(m.group(1)), train_loss=float(m.group(2)), line=line)
    if m := _ITER_VAL_RE.search(line):
        return TrainProgress(step=int(m.group(1)), val_loss=float(m.group(2)), line=line)
    return None


def build_train_args(spec: LoraTrainSpec, supported: set[str]) -> tuple[list[str], list[str]]:
    """(인자 리스트, 지원되지 않아 제외한 옵션) 반환."""
    required = ["--model", "--train", "--data", "--adapter-path"]
    missing = [f for f in required if f not in supported]
    if missing:
        raise CapabilityUnsupportedError(detail=f"mlx_lm.lora 필수 옵션 미지원: {missing}")
    args = ["--model", spec.model, "--train", "--data", str(spec.data_dir), "--adapter-path", str(spec.adapter_path)]
    skipped: list[str] = []
    for field, flag in _TRAIN_FLAG_MAP.items():
        value = getattr(spec, field)
        if value is None:
            continue
        if flag in supported:
            args += [flag, str(value)]
        else:
            skipped.append(flag)
    for flag, value in spec.extra.items():
        flag = flag if flag.startswith("--") else f"--{flag}"
        if flag in supported:
            args += [flag] if value is True else [flag, str(value)]
        else:
            skipped.append(flag)
    return args, skipped


class _ProcessTrainHandle(TrainHandle):
    def __init__(self, proc: asyncio.subprocess.Process, reader: asyncio.Task) -> None:
        self.proc = proc
        self.reader = reader

    async def wait(self) -> int:
        code = await self.proc.wait()
        await self.reader
        return code

    async def cancel(self, grace_s: float = 10.0) -> None:
        if self.proc.returncode is not None:
            return
        self.proc.send_signal(signal.SIGINT)  # graceful 먼저
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=grace_s)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()


class MlxProcessAdapter(LyricsEnginePort):
    name = "mlx"

    def __init__(self, python_exe: Path, base_model: str, cwd: Path, timeout_s: float = 600.0) -> None:
        self.python_exe = Path(python_exe)
        self.base_model = base_model
        self.cwd = Path(cwd)
        self.timeout_s = timeout_s
        self._train_flags: set[str] | None = None

    def _check_python(self) -> None:
        if not self.python_exe.is_file():
            raise EngineUnavailableError(
                "lyrics-ai 가상환경 Python 을 찾을 수 없습니다.", detail=str(self.python_exe)
            )

    async def _run(self, args: list[str], stdin: bytes | None = None, timeout: float | None = None) -> tuple[int, str, str]:
        self._check_python()
        proc = await asyncio.create_subprocess_exec(
            str(self.python_exe), *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=str(self.cwd),
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout or self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise EngineResponseError("Qwen 응답 시간 초과", detail=args[:2]) from None
        return proc.returncode or 0, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

    async def health(self) -> EngineHealth:
        try:
            code, out, err = await self._run(
                ["-c", "import mlx_lm, sys; print(getattr(mlx_lm, '__version__', 'unknown'))"], timeout=60
            )
        except EngineUnavailableError as exc:
            return EngineHealth(engine="qwen", ok=False, message=exc.user_message, details={"python": str(self.python_exe)})
        if code != 0:
            return EngineHealth(engine="qwen", ok=False, message="mlx_lm import 실패", details={"stderr": err[-500:]})
        return EngineHealth(engine="qwen", ok=True, message="MLX-LM 사용 가능", version=out.strip(),
                            details={"model": self.base_model, "python": str(self.python_exe)})

    async def generate(self, req: LLMRequest) -> LLMResult:
        payload = {
            "model": req.model or self.base_model,
            "adapter_path": req.adapter_path,
            "system": req.system,
            "user": req.user,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "seed": req.seed,
            "enable_thinking": req.enable_thinking,
        }
        started = time.monotonic()
        code, out, err = await self._run([str(WORKER_SCRIPT)], stdin=json.dumps(payload).encode("utf-8"))
        if code != 0:
            raise EngineResponseError("Qwen 생성 프로세스가 실패했습니다.", detail=err[-1500:])
        line = next((ln for ln in reversed(out.splitlines()) if ln.startswith(RESULT_PREFIX)), None)
        if line is None:
            raise EngineResponseError("Qwen 워커 결과를 해석할 수 없습니다.", detail=out[-500:])
        data = json.loads(line[len(RESULT_PREFIX):])
        return LLMResult(
            text=data.get("text", ""), thinking=data.get("thinking", ""), model=data.get("model", ""),
            adapter_path=data.get("adapter_path"), elapsed_s=time.monotonic() - started, raw=data,
        )

    async def training_options(self) -> set[str]:
        if self._train_flags is None:
            code, out, err = await self._run(["-m", "mlx_lm.lora", "--help"], timeout=120)
            if code != 0:
                raise EngineUnavailableError("mlx_lm.lora --help 실행 실패", detail=err[-500:])
            self._train_flags = parse_help_flags(out)
        return self._train_flags

    async def start_training(self, spec: LoraTrainSpec, on_progress: ProgressCallback) -> TrainHandle:
        supported = await self.training_options()
        args, skipped = build_train_args(spec, supported)
        if skipped:
            log.warning("mlx_lm.lora 미지원 옵션 제외: %s", skipped)
        self._check_python()
        proc = await asyncio.create_subprocess_exec(
            str(self.python_exe), "-m", "mlx_lm.lora", *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, cwd=str(self.cwd),
        )

        async def _pump() -> None:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip()
                progress = parse_train_line(line) or TrainProgress(line=line)
                progress.total = spec.iters
                result = on_progress(progress)
                if inspect.isawaitable(result):
                    await result

        return _ProcessTrainHandle(proc, asyncio.create_task(_pump()))
