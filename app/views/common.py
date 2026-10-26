"""View 공통: 레이아웃, VM↔View 바인딩 스코프, 상태 배지."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from nicegui import ui

from app.domain.enums import JobStatus
from app.viewmodels.base import BaseViewModel

log = logging.getLogger(__name__)

NAV = [
    ("/", "대시보드", "dashboard"),
    ("/create", "곡 만들기", "music_note"),
    ("/lyrics", "가사집", "library_books"),
    ("/library", "만든 곡", "library_music"),
    ("/music-training", "음악 LoRA 학습", "graphic_eq"),
    ("/training", "가사 LoRA 학습", "model_training"),
    ("/jobs", "작업", "pending_actions"),
    ("/settings", "설정·진단", "settings"),
]

STATUS_COLOR = {
    JobStatus.SUCCEEDED: "positive", JobStatus.FAILED: "negative", JobStatus.CANCELLED: "grey",
    JobStatus.RUNNING: "primary", JobStatus.QUEUED: "warning", JobStatus.VALIDATING: "warning",
    JobStatus.CREATED: "grey", JobStatus.EXPORTING: "primary", JobStatus.VALIDATING_ADAPTER: "primary",
}


class ViewScope:
    """한 페이지(클라이언트)의 VM 바인딩 범위.

    - ``on(prop, fn)``: VM 속성이 바뀌면 이 클라이언트 컨텍스트에서 ``fn(value)`` 실행
    - VM 의 error/info 를 토스트로 자동 표시
    - 브라우저 탭이 닫히면 VM.dispose() 로 EventBus 구독 해제 (메모리 누수 방지)
    """

    def __init__(self, vm: BaseViewModel) -> None:
        self.vm = vm
        self.client = ui.context.client
        self.client.on_delete(vm.dispose)
        self.on("error", lambda msg: msg and self.notify(msg, "negative"))
        self.on("info", lambda msg: msg and self.notify(msg, "positive"))

    def on(self, prop: str, fn: Callable[[Any], Any]) -> None:
        def _handler(value: Any) -> None:
            if self.client._deleted:  # noqa: SLF001
                return
            try:
                with self.client:
                    fn(value)
            except Exception:  # noqa: BLE001
                log.exception("View 갱신 실패 (%s)", prop)

        self.vm.subscribe(prop, _handler)

    def refresh_on(self, refreshable: Any, *props: str) -> None:
        for prop in props:
            self.on(prop, lambda _v: refreshable.refresh())

    def notify(self, message: str, kind: str = "info") -> None:
        ui.notify(message, type=kind, position="top-right", multi_line=True)
        if kind != "negative":
            self.vm.info = ""


@contextmanager
def page_frame(title: str) -> Iterator[None]:
    ui.colors(primary="#3b5bdb", secondary="#495057")
    with ui.header().classes("items-center bg-white text-black shadow-sm"):
        ui.icon("graphic_eq").classes("text-2xl text-primary")
        ui.label("AI Music Studio").classes("text-lg font-bold")
        ui.label(f"· {title}").classes("text-gray-500")
    with ui.left_drawer(value=True).classes("bg-gray-50").props("width=210"):
        for path, label, icon in NAV:
            with ui.link(target=path).classes("no-underline text-black w-full"):
                with ui.row().classes("items-center gap-2 px-2 py-1 hover:bg-gray-200 rounded w-full"):
                    ui.icon(icon).classes("text-gray-600")
                    ui.label(label)
    with ui.column().classes("w-full max-w-6xl mx-auto p-4 gap-4"):
        yield


def status_badge(status: JobStatus) -> None:
    ui.badge(status.label_ko, color=STATUS_COLOR.get(status, "grey"))


def progress_text(progress: float | None) -> str:
    return "진행률 알 수 없음" if progress is None else f"{progress * 100:.0f}%"


def section(title: str, subtitle: str = "") -> ui.card:
    card = ui.card().classes("w-full")
    with card:
        ui.label(title).classes("text-base font-bold")
        if subtitle:
            ui.label(subtitle).classes("text-xs text-gray-500")
    return card


def folder_picker(browse: Callable[[str | None], tuple[Any, list[tuple[str, bool]]]], root: Any,
                  target: Any, key: str, *, title: str = "폴더 선택", start: str | None = None) -> None:
    """서버(이 Mac)의 폴더를 탐색해 선택한 경로를 ``setattr/dict`` 로 target[key] 에 넣는 창.

    browse(path) -> (현재 폴더, [(하위 폴더 이름, 표시 여부)]) 는 ViewModel 이 제공한다.
    """
    state: dict[str, str | None] = {"path": start}

    def _set(value: str) -> None:
        if isinstance(target, dict):
            target[key] = value
        else:
            setattr(target, key, value)

    with ui.dialog() as dialog, ui.card().classes("w-[640px] max-w-full"):
        ui.label(title).classes("font-bold")

        @ui.refreshable
        def body() -> None:
            current, entries = browse(state["path"])
            state["path"] = str(current)
            with ui.row().classes("items-center gap-2 w-full no-wrap"):
                up = ui.button(icon="arrow_upward", on_click=lambda: _go(str(current.parent))).props("flat dense round")
                up.set_enabled(str(current) != str(root.resolve()))
                ui.label(str(current)).classes("text-sm font-mono break-all flex-1")
            with ui.column().classes("w-full gap-0 max-h-[50vh] overflow-y-auto border rounded"):
                if not entries:
                    ui.label("하위 폴더가 없습니다.").classes("text-xs text-gray-500 p-2")
                for name, marked in entries:
                    with ui.item(on_click=lambda n=name: _go(str(current / n))).classes("w-full"):
                        with ui.item_section().props("avatar"):
                            ui.icon("check_circle" if marked else "folder", color="positive" if marked else "grey")
                        with ui.item_section():
                            ui.item_label(name)
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("취소", on_click=dialog.close).props("flat")
                ui.button("이 폴더 선택", on_click=_choose).props("color=primary")

        def _go(path: str) -> None:
            state["path"] = path
            body.refresh()

        def _choose() -> None:
            _set(state["path"] or "")
            dialog.close()

        body()
    dialog.open()
