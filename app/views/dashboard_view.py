from __future__ import annotations

from nicegui import ui

from app.adapters.ports import EngineHealth
from app.viewmodels.dashboard_vm import DashboardViewModel
from app.views.common import ViewScope, page_frame, progress_text, section, status_badge


def _engine_card(title: str, health: EngineHealth | None) -> None:
    with ui.card().classes("min-w-[260px] flex-1"):
        ui.label(title).classes("font-bold")
        if health is None:
            ui.label("확인 중…").classes("text-gray-500")
            return
        color = "positive" if health.ok else "negative"
        with ui.row().classes("items-center gap-2"):
            ui.icon("check_circle" if health.ok else "error").props(f"color={color}")
            ui.label(health.message)
        if health.version:
            ui.label(f"버전: {health.version}").classes("text-xs text-gray-500")
        if health.is_mock:
            ui.badge("MOCK — 실제 AI 결과 아님", color="warning")


@ui.page("/")
async def dashboard_page() -> None:
    vm = DashboardViewModel()
    scope = ViewScope(vm)
    with page_frame("대시보드"):

        @ui.refreshable
        def engines() -> None:
            with ui.row().classes("w-full gap-4"):
                _engine_card("ACE-Step (음악)", vm.ace_health)
                _engine_card("Qwen / MLX-LM (가사)", vm.qwen_health)
                with ui.card().classes("min-w-[200px]"):
                    ui.label("GPU 작업 슬롯").classes("font-bold")
                    if vm.gpu_holders:
                        for desc in vm.gpu_holders.values():
                            ui.label(f"사용 중: {desc}").classes("text-primary")
                    else:
                        ui.label("유휴")
                    ui.label(f"카테고리 {vm.category_count}개").classes("text-xs text-gray-500")

        engines()
        scope.refresh_on(engines, "ace_health", "qwen_health", "gpu_holders", "category_count")
        ui.button("엔진 상태 새로고침", icon="refresh", on_click=vm.refresh_health).props("outline")

        with section("최근 작업", "학습/생성은 별도 프로세스에서 계속 진행되며, 화면을 닫아도 중단되지 않습니다."):
            @ui.refreshable
            def jobs() -> None:
                if not vm.jobs:
                    ui.label("작업이 없습니다.").classes("text-gray-500")
                for job in vm.jobs:
                    with ui.row().classes("items-center gap-3 w-full"):
                        status_badge(job.status)
                        ui.label(job.kind.value).classes("w-40")
                        ui.label(progress_text(job.progress)).classes("w-32 text-xs")
                        ui.label(job.error or "").classes("text-xs text-negative truncate flex-1")
                        if not job.status.is_terminal:
                            ui.button("취소", on_click=lambda j=job: vm.cancel_job(j.id)).props("flat dense color=negative")

            jobs()
            scope.refresh_on(jobs, "jobs")

        with section("최근 곡"):
            @ui.refreshable
            def songs() -> None:
                if not vm.recent_songs:
                    ui.label("아직 생성된 곡이 없습니다.").classes("text-gray-500")
                for song in vm.recent_songs:
                    with ui.row().classes("items-center gap-3 w-full"):
                        ui.label(song.title).classes("w-60 truncate")
                        ui.audio(f"/media/songs/{song.id}").classes("flex-1")
                        if song.generation.get("is_mock"):
                            ui.badge("MOCK", color="warning")

            songs()
            scope.refresh_on(songs, "recent_songs")

    await vm.refresh_health()
