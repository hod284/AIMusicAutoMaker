"""Finder launcher: reuse running services, otherwise start them in the background."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import os
import signal
import sqlite3
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.core.config import get_settings


def responds(url: str, key: str = "") -> bool:
    try:
        request = Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
        with urlopen(request, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def port_busy(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def start(command: list[str], cwd: Path, log: Path, env: dict[str, str]) -> subprocess.Popen:
    if not Path(command[0]).is_file():
        raise RuntimeError(f"실행 파일이 없습니다: {command[0]}")
    with log.open("ab") as stream:
        return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=stream, start_new_session=True)


def active_jobs(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return conn.execute("SELECT count(*) FROM jobs WHERE status NOT IN "
                            "('succeeded','failed','cancelled')").fetchone()[0]


def stop_service(name: str, host: str, port: int, expected_cwd: Path, marker: str, db_path: Path,
                 wait_s: float = 20) -> bool:
    """포트를 쓰는 프로그램이 정말 그 서비스인지(실행 폴더·명령) 확인한 뒤에만 끈다. 끈 경우 True."""
    if not port_busy(host, port):
        return False
    result = subprocess.run(["/usr/sbin/lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                            capture_output=True, text=True, check=False)
    pids = set(result.stdout.split())
    if len(pids) != 1:
        raise RuntimeError(f"{name} 프로세스를 확인하지 못했습니다. 자동 종료하지 않습니다.")
    pid = int(next(iter(pids)))
    cwd = subprocess.run(["/usr/sbin/lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
                         capture_output=True, text=True, check=False)
    command = subprocess.run(["/bin/ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, check=False)
    if f"n{expected_cwd}" not in cwd.stdout.splitlines() or marker not in command.stdout:
        raise RuntimeError(f"포트 {port}의 프로그램이 {name}인지 확인되지 않아 종료하지 않습니다.")
    if active_jobs(db_path):
        raise RuntimeError("진행 중인 작업이 있습니다. 작업이 끝난 뒤 다시 하세요.")
    print(f"{name}을(를) 종료합니다…", flush=True)
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + wait_s
    while port_busy(host, port):
        if time.monotonic() > deadline:
            print(f"{name}이(가) 응답하지 않아 강제 종료합니다.", flush=True)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(1)
            break
        time.sleep(0.25)
    return True


def restart_studio(host: str, port: int, db_path: Path) -> None:
    if stop_service("Studio", host, port, ROOT, "main.py", db_path):
        print("Studio를 다시 시작합니다…", flush=True)


def stop_ace(settings) -> bool:  # type: ignore[no-untyped-def]
    address = urlsplit(settings.ace_base_url)
    if address.hostname not in {"127.0.0.1", "localhost"}:
        return False
    return stop_service("음악 엔진(ACE)", address.hostname, address.port or 80, Path(settings.ace_root),
                        "acestep", settings.db_path, wait_s=30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--restart", action="store_true", help="Studio 만 껐다 켜기")
    parser.add_argument("--restart-ace", action="store_true",
                        help="음악 엔진(ACE)도 껐다 켜기 (출력이 data/launcher/ace.log 에 남음)")
    parser.add_argument("--stop", action="store_true", help="Studio 와 음악 엔진(ACE) 끄기")
    parser.add_argument("--check", action="store_true", help="설정과 현재 실행 상태만 확인")
    args = parser.parse_args()
    settings = get_settings()
    host = settings.ui_host
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    ui_url = f"http://{browser_host}:{settings.ui_port}"
    ace_url = settings.ace_base_url.rstrip("/")
    if args.check:
        print(f"Studio: {ui_url} ({'실행 중' if responds(ui_url) else '꺼짐'})")
        print(f"음악 엔진: {ace_url} ({'실행 중' if responds(ace_url + '/health', settings.ace_api_key) else '꺼짐'})")
        print(f"보컬 엔진: {settings.applio_root}")
        return
    logs = settings.data_dir / "launcher"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "launch.lock").open("a") as lock:
        # Serialise double clicks without interrupting any running service.
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.stop:
            stopped = stop_service("Studio", browser_host, settings.ui_port, ROOT, "main.py", settings.db_path)
            stopped = stop_ace(settings) or stopped
            print("모두 종료했습니다." if stopped else "실행 중인 Studio·음악 엔진이 없습니다.", flush=True)
            return
        if args.restart or args.restart_ace:
            restart_studio(browser_host, settings.ui_port, settings.db_path)
        if args.restart_ace and stop_ace(settings):
            print("음악 엔진(ACE)을 다시 시작합니다…", flush=True)
        env = settings.ace_subprocess_env()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        ace_process = None
        if settings.music_engine == "ace" and not responds(ace_url + "/health", settings.ace_api_key):
            address = urlsplit(ace_url)
            if address.hostname in {"127.0.0.1", "localhost"}:
                port = address.port or 80
                if port_busy(address.hostname, port):
                    print("음악 엔진이 시작 중이거나 포트가 사용 중입니다. 기존 프로세스를 유지합니다.", flush=True)
                else:
                    env.setdefault("ACESTEP_LM_BACKEND", "mlx")
                    command = [str(settings.ace_python), "-m", "acestep.api_server",
                               "--host", "127.0.0.1", "--port", str(port)]
                    if settings.ace_api_key:
                        command += ["--api-key", settings.ace_api_key]
                    ace_process = start(command, settings.ace_root, logs / "ace.log", env)
                    print("음악 엔진을 시작합니다. 첫 실행은 모델을 읽는 데 시간이 걸릴 수 있어요.", flush=True)
            else:
                print("외부 음악 서버에 연결할 수 없습니다. Studio는 먼저 실행합니다.", flush=True)
        studio_process = None
        if not responds(ui_url):
            if port_busy(browser_host, settings.ui_port):
                raise RuntimeError(f"Studio 포트 {settings.ui_port}를 다른 프로그램이 사용 중입니다.")
            studio_process = start([str(ROOT / ".venv/bin/python"), str(ROOT / "main.py")],
                                   ROOT, logs / "studio.log", env)
            print("Studio를 시작합니다…", flush=True)
        deadline = time.monotonic() + 90
        while not responds(ui_url):
            if studio_process is not None and studio_process.poll() is not None:
                raise RuntimeError(f"Studio 실행에 실패했습니다. 로그: {logs / 'studio.log'}")
            if time.monotonic() > deadline:
                raise RuntimeError(f"Studio 시작 시간이 초과됐습니다. 로그: {logs / 'studio.log'}")
            time.sleep(0.5)
        if not args.no_browser:
            opened = False
            for command in (["/usr/bin/open", ui_url],
                            ["/usr/bin/open", "-a", "Safari", ui_url],
                            ["/usr/bin/open", "-a", "Google Chrome", ui_url]):
                result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode == 0:
                    opened = True
                    break
            if not opened:
                print(f"브라우저 자동 열기를 사용할 수 없습니다. 다음 주소를 직접 여세요: {ui_url}", flush=True)
        print(f"Studio 준비 완료: {ui_url}", flush=True)
        if ace_process is not None:
            print(f"음악 엔진은 백그라운드에서 준비됩니다. 로그: {logs / 'ace.log'}", flush=True)
        print("이 터미널 창을 닫아도 Studio는 계속 실행됩니다.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"실행 오류: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
