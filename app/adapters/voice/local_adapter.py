"""로컬 목소리 엔진 어댑터: audio-separator(분리) + Seed-VC(변환·학습) + FFmpeg(자르기·합치기).

- 모두 별도 가상환경의 명령줄 도구를 하위 프로세스로 실행한다 (``shell=True`` 금지, create_subprocess_exec).
- 옵션은 설치본 코드에서 확인한 것만 쓴다:
  * audio-separator 0.47.1 ``utils/cli.py``: audio_files, --model_file_dir, --output_dir, --output_format,
    --custom_output_names, -m/--model_filename
  * Seed-VC (Plachtaa/seed-vc) ``inference.py``: --source --target --output --diffusion-steps --length-adjust
    --inference-cfg-rate --f0-condition --auto-f0-adjust --semi-tone-shift --checkpoint --config
  * Seed-VC ``train.py``: --config --dataset-dir --run-name --batch-size --max-steps --max-epochs --save-every
    --num-workers  (결과: <config log_dir>/<run_name>/ft_model.pth + 설정 yml 복사본)
- Seed-VC 는 상대 경로(./checkpoints, ./modules/...)를 쓰므로 반드시 seed-vc 폴더에서 실행한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
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
from app.core.errors import EngineResponseError, EngineUnavailableError, ValidationError

log = logging.getLogger(__name__)

_STEP_RE = re.compile(r"epoch\s+(\d+),\s*step\s+(\d+),\s*loss:\s*([-+0-9.eE]+|nan|inf)", re.I)
_SEP_NAMES = {"Vocals": "vocals", "Instrumental": "instrumental"}
_CODEC = {".mp3": ["-c:a", "libmp3lame", "-b:a", "192k"], ".wav": ["-c:a", "pcm_s16le"], ".flac": ["-c:a", "flac"]}


def parse_seedvc_train_line(line: str) -> TrainProgress | None:
    """Seed-VC train.py 의 ``epoch E, step S, loss: L`` 줄."""
    m = _STEP_RE.search(line)
    if not m:
        return None
    try:
        loss = float(m.group(3))
    except ValueError:
        loss = None
    return TrainProgress(step=int(m.group(2)), train_loss=loss, line=line)


def last_error(lines: list[str]) -> str:
    """하위 프로세스 출력에서 실제 오류 줄 (예: 'TypeError: ...'). 없으면 마지막 줄."""
    for line in reversed(lines):
        text = line.strip()
        if re.match(r"^[A-Za-z_.]*(Error|Exception|Interrupt)\b.*:", text) or text.startswith("RuntimeError"):
            return text[:300]
    return (lines[-1].strip()[:300] if lines else "")


def wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except (wave.Error, OSError, EOFError):
        return 0.0


def _fresh_dir(path: Path) -> Path:
    if path.exists() and any(path.iterdir()):
        raise ValidationError("작업 폴더가 비어 있지 않습니다 (덮어쓰지 않음).", detail=str(path))
    path.mkdir(parents=True, exist_ok=True)
    return path


async def _iter_lines(stream: asyncio.StreamReader):  # type: ignore[no-untyped-def]
    """tqdm 의 \\r 진행 표시도 줄로 나눠 읽는다."""
    buf = b""
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        buf += chunk
        parts = re.split(rb"[\r\n]", buf)
        buf = parts.pop()
        for p in parts:
            if p.strip():
                yield p.decode("utf-8", "replace").rstrip()
    if buf.strip():
        yield buf.decode("utf-8", "replace").rstrip()


class _ProcHandle(TrainHandle):
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
        self.proc.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(self.proc.wait(), timeout=grace_s)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()


class LocalVoiceAdapter(VoiceEnginePort):
    name = "local"

    def __init__(self, separator_bin: Path, separator_model_dir: Path, seedvc_root: Path, seedvc_python: Path,
                 ffmpeg: str = "ffmpeg", separator_model: str = "", timeout_s: float = 3600.0,
                 env: dict[str, str] | None = None, applio_root: Path | None = None,
                 applio_python: Path | None = None) -> None:
        self.separator_bin = Path(separator_bin)
        self.separator_model_dir = Path(separator_model_dir)
        self.seedvc_root = Path(seedvc_root)
        self.seedvc_python = Path(seedvc_python)
        self.applio_root = Path(applio_root) if applio_root else None
        self.applio_python = Path(applio_python) if applio_python else None
        self.ffmpeg = ffmpeg
        self.separator_model = separator_model
        self.timeout_s = timeout_s
        self.env = env or dict(os.environ)

    # --- 공통 실행 ---
    async def _run(self, exe: str, args: list[str], cwd: Path | None, log_fn: LogFn | None,
                   timeout: float | None = None) -> tuple[int, list[str]]:
        try:
            proc = await asyncio.create_subprocess_exec(
                exe, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                stdin=asyncio.subprocess.DEVNULL, cwd=str(cwd) if cwd else None, env=self.env,
            )
        except FileNotFoundError as exc:
            raise EngineUnavailableError("실행 파일을 찾을 수 없습니다.", detail=exe) from exc
        tail: list[str] = []

        async def _pump() -> None:
            assert proc.stdout is not None
            async for line in _iter_lines(proc.stdout):
                tail.append(line)
                del tail[:-80]
                if log_fn:
                    log_fn(line)

        reader = asyncio.create_task(_pump())
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout or self.timeout_s)
            await reader
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise EngineResponseError("목소리 작업 시간 초과", detail=" ".join([Path(exe).name, *args[:2]])) from None
        except BaseException:          # 취소되면 하위 프로세스도 같이 끝낸다
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        return proc.returncode or 0, tail

    def _need(self, path: Path, what: str) -> None:
        if not path.exists():
            raise EngineUnavailableError(f"{what} 을(를) 찾을 수 없습니다. 설치 상태를 확인하세요.", detail=str(path))

    # --- 상태 ---
    async def health(self) -> EngineHealth:
        checks = {
            "audio-separator": self.separator_bin.is_file(),
            "seed-vc python": self.seedvc_python.is_file(),
            "seed-vc inference.py": (self.seedvc_root / "inference.py").is_file(),
            "seed-vc train.py": (self.seedvc_root / "train.py").is_file(),
            "ffmpeg": bool(shutil.which(self.ffmpeg)) or Path(self.ffmpeg).is_file(),
        }
        missing = [k for k, ok in checks.items() if not ok]
        return EngineHealth(
            engine="voice", ok=not missing,
            message="보컬 분리·목소리 변환 사용 가능" if not missing else f"설치 필요: {', '.join(missing)}",
            details={"separator": str(self.separator_bin), "seedvc": str(self.seedvc_root), **checks},
        )

    # --- 분리 (audio-separator) ---
    async def separate(self, src: Path, out_dir: Path, log: LogFn) -> tuple[Path, Path]:
        self._need(self.separator_bin, "audio-separator")
        out = _fresh_dir(Path(out_dir))
        self.separator_model_dir.mkdir(parents=True, exist_ok=True)
        args = [str(src), "--model_file_dir", str(self.separator_model_dir), "--output_dir", str(out),
                "--output_format", "WAV", "--custom_output_names", json.dumps(_SEP_NAMES)]
        if self.separator_model:
            args += ["--model_filename", self.separator_model]
        log("보컬/반주 분리 시작 (audio-separator)")
        code, tail = await self._run(str(self.separator_bin), args, None, log)
        if code != 0:
            raise EngineResponseError(f"보컬 분리 실패 (종료 코드 {code}): {last_error(tail)}",
                                      detail="\n".join(tail[-15:]))
        files = [p for p in out.iterdir() if p.is_file()]
        vocals = next((p for p in files if p.stem.lower() == "vocals"), None) or \
            next((p for p in files if "vocals" in p.name.lower()), None)
        inst = next((p for p in files if p.stem.lower() == "instrumental"), None) or \
            next((p for p in files if "instrumental" in p.name.lower()), None)
        if vocals is None or inst is None:
            raise EngineResponseError("분리 결과 파일(보컬/반주)을 찾지 못했습니다.", detail=str([p.name for p in files]))
        return vocals, inst

    # --- 변환 (Seed-VC inference.py) ---
    async def convert(self, spec: VoiceConvertSpec, log: LogFn) -> Path:
        if spec.backend == "rvc":
            converted = await self._convert_rvc(spec, log)
            return await self.match_vocal_level(spec.source, converted, log)
        if spec.reference is None:
            raise ValidationError("Seed-VC 참고 녹음이 없습니다.")
        self._need(self.seedvc_python, "Seed-VC 가상환경 python")
        out = _fresh_dir(Path(spec.output_dir))
        args = ["inference.py", "--source", str(spec.source), "--target", str(spec.reference), "--output", str(out),
                "--diffusion-steps", str(spec.diffusion_steps), "--length-adjust", "1.0",
                "--inference-cfg-rate", str(spec.inference_cfg_rate),
                "--f0-condition", "True" if spec.f0_condition else "False", "--auto-f0-adjust", "False",
                "--semi-tone-shift", str(spec.semi_tone_shift)]
        if spec.checkpoint is not None:
            if spec.config is None:
                raise ValidationError("목소리 모델 설정 파일(yml)이 없습니다.", detail=str(spec.checkpoint))
            args += ["--checkpoint", str(spec.checkpoint), "--config", str(spec.config)]
        log("목소리 변환 시작 (Seed-VC)")
        code, tail = await self._run(str(self.seedvc_python), args, self.seedvc_root, log)
        if code != 0:
            raise EngineResponseError(f"목소리 변환 실패 (종료 코드 {code}): {last_error(tail)}",
                                      detail="\n".join(tail[-15:]))
        results = sorted(out.glob("vc_*.wav"), key=lambda p: p.stat().st_mtime)
        if not results:
            raise EngineResponseError("변환 결과(vc_*.wav)를 찾지 못했습니다.", detail="\n".join(tail[-10:]))
        return await self.match_vocal_level(spec.source, results[-1], log)

    async def vocal_loudness(self, path: Path) -> float:
        """Measure gated loudness in the same stereo layout used by mix()."""
        args = ["-hide_banner", "-nostdin", "-nostats", "-i", str(path), "-vn", "-af",
                "aresample=44100,aformat=channel_layouts=stereo,"
                "ebur128=peak=true:framelog=verbose", "-f", "null", "-"]
        code, tail = await self._run(self.ffmpeg, args, None, None, timeout=600)
        values = re.findall(r"\bI:\s*(-?\d+(?:\.\d+)?)\s+LUFS", "\n".join(tail))
        if code != 0 or not values:
            raise EngineResponseError("보컬 음량 측정 실패", detail="\n".join(tail[-15:]))
        return float(values[-1])

    async def match_vocal_level(self, reference: Path, converted: Path, log: LogFn) -> Path:
        """Restore source vocal loudness with one gain; never normalize the backing track.

        Gating excludes pauses. A fixed gain preserves the converted performance's
        dynamics; the limiter catches peaks only, with automatic gain disabled.
        """
        original_i = await self.vocal_loudness(reference)
        converted_i = await self.vocal_loudness(converted)
        if original_i <= -60:
            log("원본 보컬이 무음에 가까워 음량 보정을 생략합니다.")
            return converted
        if converted_i <= -60:
            raise EngineResponseError("변환된 보컬이 무음에 가깝습니다. 목소리 모델과 원본을 확인하세요.")
        requested = original_i - converted_i
        gain = max(-18.0, min(18.0, requested))
        log(f"보컬 음량 맞춤: 원본 {original_i:.1f} LUFS → 변환 {converted_i:.1f} LUFS, 보정 {gain:+.1f} dB")
        if abs(requested - gain) > 0.1:
            log("보컬 음량 차이가 커서 보정 폭을 ±18 dB로 제한했습니다.")
        if abs(gain) < 0.25:
            return converted
        adjusted = converted.with_name(converted.stem + "_level_matched.wav")
        args = ["-hide_banner", "-nostdin", "-loglevel", "error", "-n", "-i", str(converted), "-vn",
                "-af", "aresample=44100,aformat=channel_layouts=stereo,"
                f"volume={gain:.4f}dB,alimiter=limit=0.97:level=false:latency=true",
                "-c:a", "pcm_s16le", str(adjusted)]
        try:
            code, tail = await self._run(self.ffmpeg, args, None, None, timeout=600)
            if code != 0 or not adjusted.is_file():
                raise EngineResponseError("보컬 음량 보정 실패", detail="\n".join(tail[-15:]))
            adjusted.replace(converted)
        finally:
            adjusted.unlink(missing_ok=True)
        return converted

    def validate_rvc(self) -> None:
        if self.applio_root is None or self.applio_python is None:
            raise ValidationError("Applio 경로를 설정하세요 (STUDIO_APPLIO_ROOT / STUDIO_APPLIO_PYTHON).")
        self._need(self.applio_root / "core.py", "Applio core.py — Applio 설치 경로를 확인하세요")
        self._need(self.applio_python, "Applio 가상환경 python — Applio 설치를 완료하세요")

    async def _convert_rvc(self, spec: VoiceConvertSpec, log: LogFn) -> Path:
        self.validate_rvc()
        if spec.checkpoint is None:
            raise ValidationError("RVC 모델(.pth)이 없습니다.")
        self._need(spec.checkpoint, "RVC 모델")
        if spec.index:
            self._need(spec.index, "RVC 인덱스")
        out = _fresh_dir(Path(spec.output_dir))
        dest = out / "rvc_vocals.wav"
        worker = Path(__file__).resolve().parents[2] / "workers" / "rvc_convert_worker.py"
        args = [str(worker), "--root", str(self.applio_root), "--source", str(spec.source),
                "--output", str(dest), "--model", str(spec.checkpoint),
                "--pitch", str(spec.semi_tone_shift), "--index-rate", str(spec.index_rate)]
        if spec.index:
            args += ["--index", str(spec.index)]
        log("목소리 변환 시작 (RVC / Applio)")
        code, tail = await self._run(str(self.applio_python), args, self.applio_root, log)
        if code != 0 or not dest.is_file() or dest.stat().st_size <= 44:
            raise EngineResponseError(f"RVC 변환 실패: {last_error(tail) or '출력 음원이 없습니다.'}",
                                      detail="\n".join(tail[-15:]))
        return dest

    # --- 학습 (Seed-VC train.py) ---
    def _write_config(self, spec: VoiceTrainSpec, dest_dir: Path) -> Path:
        """프리셋을 복사하되 log_dir 만 결과 폴더의 부모로 바꾼다 (Seed-VC 원본 파일은 건드리지 않음)."""
        preset = self.seedvc_root / "configs" / "presets" / spec.preset
        self._need(preset, "Seed-VC 프리셋 설정")
        text = preset.read_text(encoding="utf-8")
        new_text, n = re.subn(r"(?m)^log_dir:.*$", f"log_dir: {json.dumps(str(spec.log_dir))}", text, count=1)
        if n != 1:
            raise EngineResponseError("Seed-VC 프리셋에서 log_dir 항목을 찾지 못했습니다.", detail=str(preset))
        dest_dir.mkdir(parents=True, exist_ok=True)
        cfg = dest_dir / spec.preset          # train.py 가 같은 이름으로 결과 폴더에 복사한다
        cfg.write_text(new_text, encoding="utf-8")
        return cfg

    async def start_training(self, spec: VoiceTrainSpec, on_progress: ProgressCallback) -> TrainHandle:
        self._need(self.seedvc_python, "Seed-VC 가상환경 python")
        self._need(self.seedvc_root / "train.py", "Seed-VC train.py")
        cfg = self._write_config(spec, spec.dataset_dir.parent / "_seedvc_config")
        args = ["train.py", "--config", str(cfg), "--dataset-dir", str(spec.dataset_dir), "--run-name", spec.run_name,
                "--batch-size", str(spec.batch_size), "--max-steps", str(spec.max_steps), "--max-epochs", "1000",
                "--save-every", str(spec.save_every), "--num-workers", "0"]
        proc = await asyncio.create_subprocess_exec(
            str(self.seedvc_python), *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, cwd=str(self.seedvc_root), env=self.env,
        )

        async def _pump() -> None:
            assert proc.stdout is not None
            async for line in _iter_lines(proc.stdout):
                tp = parse_seedvc_train_line(line) or TrainProgress(line=line)
                tp.total = spec.max_steps
                result = on_progress(tp)
                if asyncio.iscoroutine(result):
                    await result

        return _ProcHandle(proc, asyncio.create_task(_pump()))

    # --- FFmpeg ---
    async def split_clips(self, src: Path, out_dir: Path, seconds: float = 15.0) -> list[Path]:
        out = _fresh_dir(Path(out_dir))
        stem = re.sub(r"[^\w가-힣-]+", "_", Path(src).stem)[:40] or "clip"
        args = ["-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "44100",
                "-f", "segment", "-segment_time", str(seconds), "-reset_timestamps", "1", "-c:a", "pcm_s16le",
                str(out / f"{stem}_%03d.wav")]
        code, tail = await self._run(self.ffmpeg, args, None, None, timeout=600)
        if code != 0:
            raise EngineResponseError(f"오디오 자르기 실패: {Path(src).name}", detail="\n".join(tail[-10:]))
        clips = []
        for p in sorted(out.glob("*.wav")):
            if wav_seconds(p) < 1.0:     # Seed-VC 학습은 1초 미만 조각을 무시한다
                p.unlink()
            else:
                clips.append(p)
        return clips

    async def mix(self, vocals: Path, instrumental: Path, dest: Path) -> Path:
        dest = Path(dest)
        if dest.exists():
            raise ValidationError("같은 이름의 파일이 이미 있습니다 (덮어쓰지 않음).", detail=str(dest))
        codec = _CODEC.get(dest.suffix.lower())
        if codec is None:
            raise ValidationError(f"지원하지 않는 저장 형식: {dest.suffix}")
        graph = ("[0:a]aresample=44100,aformat=channel_layouts=stereo[i];"
                 "[1:a]aresample=44100,aformat=channel_layouts=stereo[v];"
                 "[i][v]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.97[out]")
        args = ["-hide_banner", "-nostdin", "-loglevel", "error", "-n", "-i", str(instrumental), "-i", str(vocals),
                "-filter_complex", graph, "-map", "[out]", *codec, str(dest)]
        code, tail = await self._run(self.ffmpeg, args, None, None, timeout=600)
        if code != 0 or not dest.is_file():
            raise EngineResponseError("보컬과 반주 합치기 실패 (FFmpeg)", detail="\n".join(tail[-10:]))
        return dest
