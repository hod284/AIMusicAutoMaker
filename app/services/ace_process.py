"""음악 엔진(ACE) 프로세스 다시 시작.

ACE 는 학습이 끝나도 PyTorch 가 잡아 둔 메모리를 비우지 않아, 같은 프로세스로 곡을 만들면
맥에서 'MPS backend out of memory' 가 난다 (2026-11-08: LoRA 10시간 학습 후 곡 생성 실패, 재시작 후 정상).
프로세스를 다시 띄우는 것이 확실한 방법이라 학습이 끝나면 앱이 직접 다시 시작한다.

안전장치 (scripts/launch_studio.py 와 같은 규칙):
- ACE 주소가 이 컴퓨터(127.0.0.1/localhost)일 때만
- 포트를 쓰는 프로세스가 정확히 하나이고, 실행 폴더가 ACE 폴더이며 명령에 'acestep' 이 있을 때만 끈다
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from app.core.config import Settings

log = logging.getLogger(__name__)

LSOF = "/usr/sbin/lsof"
PS = "/bin/ps"


class AceProcessManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        address = urlsplit(settings.ace_base_url)
        self.host = address.hostname or ""
        self.port = address.port or 80

    @property
    def enabled(self) -> bool:
        """실제 ACE 를 이 컴퓨터에서 쓰고, 프로세스를 확인할 도구(lsof)가 있을 때만."""
        return (self.settings.music_engine == "ace" and self.host in {"127.0.0.1", "localhost"}
                and Path(LSOF).is_file())

    # --- 확인 ---
    def _port_busy(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=1):
                return True
        except OSError:
            return False

    def _healthy(self) -> bool:
        key = self.settings.ace_api_key
        request = Request(self.settings.ace_base_url.rstrip("/") + "/health",
                          headers={"Authorization": f"Bearer {key}"} if key else {})
        try:
            with urlopen(request, timeout=3) as response:
                return response.status == 200
        except Exception:  # noqa: BLE001
            return False

    def _ace_pid(self) -> int:
        result = subprocess.run([LSOF, "-t", f"-iTCP:{self.port}", "-sTCP:LISTEN"],
                                capture_output=True, text=True, check=False)
        pids = set(result.stdout.split())
        if len(pids) != 1:
            raise RuntimeError(f"포트 {self.port} 의 음악 엔진 프로세스를 하나로 확인하지 못했습니다 ({len(pids)}개).")
        pid = int(next(iter(pids)))
        cwd = subprocess.run([LSOF, "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                             capture_output=True, text=True, check=False)
        command = subprocess.run([PS, "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False)
        if f"n{Path(self.settings.ace_root)}" not in cwd.stdout.splitlines() or "acestep" not in command.stdout:
            raise RuntimeError(f"포트 {self.port} 의 프로그램이 음악 엔진(ACE)인지 확인되지 않아 끄지 않습니다.")
        return pid

    # --- 끄기 / 켜기 ---
    def _stop(self, wait_s: float = 30) -> None:
        if not self._port_busy():
            return
        pid = self._ace_pid()
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + wait_s
        while self._port_busy():
            if time.monotonic() > deadline:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                time.sleep(1)
                break
            time.sleep(0.25)

    def _start(self) -> None:
        s = self.settings
        env = s.ace_subprocess_env()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("ACESTEP_LM_BACKEND", "mlx")
        command = [str(s.ace_python), "-m", "acestep.api_server", "--host", "127.0.0.1", "--port", str(self.port)]
        if s.ace_api_key:
            command += ["--api-key", s.ace_api_key]
        if not Path(command[0]).is_file():
            raise RuntimeError(f"음악 엔진 실행 파일이 없습니다: {command[0]}")
        logs = s.data_dir / "launcher"
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / "ace.log").open("ab") as stream:
            subprocess.Popen(command, cwd=s.ace_root, env=env, stdin=subprocess.DEVNULL,
                             stdout=stream, stderr=stream, start_new_session=True)

    def restart_blocking(self, health_timeout_s: float = 300, sleep: Callable[[float], None] = time.sleep) -> None:
        self._stop()
        self._start()
        deadline = time.monotonic() + health_timeout_s
        while not self._healthy():
            if time.monotonic() > deadline:
                raise RuntimeError(f"음악 엔진이 {int(health_timeout_s)}초 안에 다시 켜지지 않았습니다.")
            sleep(2)

    async def restart(self, health_timeout_s: float = 300) -> None:
        await asyncio.to_thread(self.restart_blocking, health_timeout_s)
