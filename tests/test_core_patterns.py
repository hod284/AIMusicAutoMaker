"""싱글톤 / 옵저버 / 상태머신 / 경로 보안 / 원자적 IO."""

from __future__ import annotations

import asyncio
import threading

import pytest

from app.core.errors import InvalidStateTransition, PathSecurityError
from app.core.observer import EventBus, Observable, ObservableProperty
from app.core.singleton import SingletonMeta
from app.domain.enums import JobStatus
from app.domain.job_state import ensure_transition
from app.utils.fileio import atomic_write_text, slugify
from app.utils.path_guard import PathGuard, safe_filename


class _Single(metaclass=SingletonMeta):
    def __init__(self) -> None:
        self.created = True


def test_singleton_same_instance_across_threads():
    results = []
    threads = [threading.Thread(target=lambda: results.append(_Single())) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(r) for r in results}) == 1
    _Single.reset()
    assert _Single() is not results[0]


def test_event_bus_is_singleton_and_wildcard():
    bus = EventBus()
    assert bus is EventBus()
    got = []
    sub = bus.subscribe("job.*", lambda e: got.append(e.topic))
    bus.publish("job.updated", x=1)
    bus.publish("song.created")
    assert got == ["job.updated"]
    sub.dispose()
    bus.publish("job.updated")
    assert got == ["job.updated"]


def test_event_bus_listener_error_isolated():
    bus = EventBus()
    got = []
    bus.subscribe("t", lambda e: 1 / 0)
    bus.subscribe("t", lambda e: got.append(1))
    bus.publish("t")
    assert got == [1]


def test_event_bus_async_listener():
    bus = EventBus()
    got = []

    async def listener(e):
        got.append(e.payload["v"])

    async def main():
        bus.subscribe("t", listener)
        bus.publish("t", v=5)
        await asyncio.sleep(0)

    asyncio.run(main())
    assert got == [5]


class _VM(Observable):
    name = ObservableProperty("")
    items = ObservableProperty(factory=list)


def test_observable_property_notifies_only_on_change():
    vm = _VM()
    seen = []
    vm.subscribe("name", seen.append)
    vm.name = "a"
    vm.name = "a"
    vm.name = "b"
    assert seen == ["a", "b"]
    all_seen = []
    vm.subscribe("*", lambda n, v: all_seen.append(n))
    vm.items = [1]
    vm.items = [1]  # 리스트 재할당은 항상 알림
    assert all_seen == ["items", "items"]


def test_state_machine():
    ensure_transition(JobStatus.CREATED, JobStatus.VALIDATING)
    ensure_transition(JobStatus.RUNNING, JobStatus.EXPORTING)
    with pytest.raises(InvalidStateTransition):
        ensure_transition(JobStatus.SUCCEEDED, JobStatus.RUNNING)
    with pytest.raises(InvalidStateTransition):
        ensure_transition(JobStatus.CREATED, JobStatus.SUCCEEDED)


def test_path_guard_blocks_traversal_symlink_and_protected(tmp_path):
    root = tmp_path / "ws"
    root.mkdir()
    protected = tmp_path / "ws" / "keep"
    protected.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "link").symlink_to(outside)
    guard = PathGuard([root], protected=[protected])
    assert guard.resolve("a/b", base=root) == root / "a" / "b"
    with pytest.raises(PathSecurityError):
        guard.resolve("../outside", base=root)
    with pytest.raises(PathSecurityError):
        guard.resolve(root / "link" / "x")
    with pytest.raises(PathSecurityError):
        guard.ensure_writable(protected / "file")
    with pytest.raises(PathSecurityError):
        guard.child(root, "..")


def test_safe_filename_and_slug():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("내 노래 (final).MP3") == "내_노래_final.mp3"
    assert slugify("2000년대 감성 Rock Ballad!") == "2000년대-감성-rock-ballad"


def test_atomic_write_never_overwrites(tmp_path):
    p = tmp_path / "a.json"
    atomic_write_text(p, "1")
    with pytest.raises(FileExistsError):
        atomic_write_text(p, "2")
    assert p.read_text() == "1"
    atomic_write_text(p, "3", overwrite=True)
    assert p.read_text() == "3"
    assert not [f for f in tmp_path.iterdir() if f.name.endswith(".tmp")]
