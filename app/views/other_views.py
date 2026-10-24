"""Library / Jobs / Training / Settings 페이지."""

from __future__ import annotations

from nicegui import ui

from app.adapters.ports import CapabilityState
from app.viewmodels.misc_vms import JobsViewModel, LibraryViewModel, SettingsViewModel, TrainingViewModel
from app.views.common import ViewScope, page_frame, progress_text, section, status_badge

_CAP_COLOR = {CapabilityState.AVAILABLE: "positive", CapabilityState.MOCK: "warning",
              CapabilityState.MISSING: "negative", CapabilityState.UNBOUND: "grey", CapabilityState.UNKNOWN: "grey"}


@ui.page("/library")
def library_page() -> None:
    vm = LibraryViewModel()
    scope = ViewScope(vm)
    with page_frame("곡 라이브러리"):
        with ui.row().classes("items-end gap-3"):
            options = {"": "전체"} | {c.id: c.name for c in vm.categories}
            ui.select(options, value="", label="카테고리", on_change=lambda e: vm.set_filter(category_id=e.value or "")).classes("w-60")
            ui.input("검색 (제목/프롬프트)", on_change=lambda e: vm.set_filter(query=e.value or "")).classes("w-72")

        @ui.refreshable
        def listing() -> None:
            if not vm.songs:
                ui.label("곡이 없습니다.").classes("text-gray-500")
            for song in vm.songs:
                g = song.generation
                with ui.card().classes("w-full"):
                    with ui.row().classes("items-center gap-2"):
                        ui.label(song.title).classes("font-bold")
                        ui.badge(vm.category_name(song.category_id))
                        if g.get("is_mock"):
                            ui.badge("MOCK", color="warning")
                        ui.label(song.created_at.strftime("%Y-%m-%d %H:%M")).classes("text-xs text-gray-500")
                    ui.audio(f"/media/songs/{song.id}").classes("w-full")
                    with ui.expansion("가사 · 프롬프트 · 모델 · Seed").classes("w-full"):
                        ui.label(f"프롬프트: {song.prompt}").classes("text-sm")
                        lora = g.get("lora", {})
                        ui.label(f"음악 LoRA: {lora.get('adapter_path') or '기본 모델'} ({lora.get('load_status', '-')})").classes("text-xs")
                        ui.label(f"Seed: {g.get('seed', '-')} · DiT: {g.get('dit_model', '-')} · task: {g.get('task_id', '-')}").classes("text-xs")
                        ui.markdown(f"```\n{g.get('lyrics', '')}\n```")
                    ui.link("다운로드", f"/media/songs/{song.id}?download=1")

        listing()
        scope.refresh_on(listing, "songs")


@ui.page("/jobs")
def jobs_page() -> None:
    vm = JobsViewModel()
    scope = ViewScope(vm)
    with page_frame("작업"):
        with ui.row().classes("w-full gap-4 items-start no-wrap"):
            @ui.refreshable
            def listing() -> None:
                with ui.column().classes("w-[460px] shrink-0 gap-1"):
                    for job in vm.jobs:
                        with ui.item(on_click=lambda j=job: vm.select(j.id)).classes(
                                "rounded w-full " + ("bg-blue-50" if job.id == vm.selected_id else "")):
                            with ui.row().classes("items-center gap-2 w-full"):
                                status_badge(job.status)
                                ui.label(job.kind.value).classes("w-36")
                                ui.label(progress_text(job.progress)).classes("text-xs w-28")
                                ui.label(job.created_at.strftime("%m-%d %H:%M:%S")).classes("text-xs text-gray-500")
                                if not job.status.is_terminal:
                                    ui.button(icon="stop", on_click=lambda j=job: vm.cancel(j.id)).props("flat dense round color=negative")

            listing()
            scope.refresh_on(listing, "jobs", "selected_id")

            @ui.refreshable
            def events() -> None:
                with ui.card().classes("flex-1"):
                    ui.label("로그").classes("font-bold")
                    if not vm.selected_id:
                        ui.label("작업을 선택하세요.").classes("text-gray-500")
                        return
                    job = next((j for j in vm.jobs if j.id == vm.selected_id), None)
                    if job and job.error:
                        ui.label(job.error).classes("text-negative")
                    log = ui.log(max_lines=1000).classes("w-full h-[60vh] text-xs")
                    for e in vm.events:
                        log.push(f"{e.timestamp.strftime('%H:%M:%S')} [{e.level}] {e.message}")

            events()
            scope.refresh_on(events, "events", "selected_id")


@ui.page("/training")
async def training_page() -> None:
    vm = TrainingViewModel()
    scope = ViewScope(vm)
    with page_frame("가사 LoRA 학습"):
        ui.select({c.id: c.name for c in vm.categories}, label="카테고리", value=vm.category_id,
                  on_change=lambda e: vm.select(e.value)).classes("w-72")

        with section("가사 LoRA 데이터셋 (MLX-LM chat JSONL)",
                     '한 줄에 {"messages":[{"role":"user",...},{"role":"assistant",...}]}. 권리가 확인된 가사만 사용하세요.'):
            text = {"value": ""}

            async def on_upload(e) -> None:  # type: ignore[no-untyped-def]
                text["value"] = await e.file.text()
                vm.preview_jsonl(text["value"])

            ui.upload(label="JSONL 업로드", on_upload=on_upload, auto_upload=True, max_file_size=20_000_000).props(
                "accept=.jsonl")
            ui.textarea("또는 붙여넣기").bind_value(text, "value").classes("w-full font-mono")
            with ui.row():
                ui.button("검증", on_click=lambda: vm.preview_jsonl(text["value"])).props("outline")
                ui.button("데이터셋 버전 만들기 (train/valid)", on_click=lambda: vm.build_dataset(text["value"]))

            @ui.refreshable
            def report() -> None:
                r = vm.jsonl_report
                if r:
                    ui.label(f"전체 {r.total} · 유효 {r.valid} · 중복 {r.duplicates} · 오류 {len(r.errors)}")
                    for msg in r.errors[:10] + r.warnings:
                        ui.label(msg).classes("text-xs text-orange-700")

            report()
            scope.refresh_on(report, "jsonl_report")

        with section("가사 LoRA 학습", "mlx_lm.lora --help 로 확인된 옵션만 전달합니다. 결과는 새 버전 폴더에 저장됩니다."):
            cfg = {"dataset_id": None, "iters": 100, "batch_size": 1, "learning_rate": None}

            @ui.refreshable
            def train_form() -> None:
                ui.select({d.id: f"v{d.version} ({d.source_path})" for d in vm.datasets}, label="데이터셋").bind_value(
                    cfg, "dataset_id").classes("w-full")

            train_form()
            scope.refresh_on(train_form, "datasets")
            with ui.row().classes("items-end gap-3"):
                ui.number("iters", min=1).bind_value(cfg, "iters").classes("w-28")
                ui.number("batch size", min=1).bind_value(cfg, "batch_size").classes("w-28")
                ui.number("learning rate (빈칸=기본)", format="%.6f").bind_value(cfg, "learning_rate").classes("w-48")
                ui.button("학습 시작", icon="play_arrow", on_click=lambda: cfg["dataset_id"] and vm.start_lyrics_training(
                    cfg["dataset_id"], int(cfg["iters"]), int(cfg["batch_size"]), cfg["learning_rate"]))
                ui.button("중지", icon="stop", on_click=vm.cancel).props("outline color=negative")
                ui.button("CLI 옵션 확인 (--help)", on_click=vm.load_train_options).props("flat")
            opts = ui.label("").classes("text-xs text-gray-500")
            scope.on("train_options", lambda v: opts.set_text("지원 옵션: " + " ".join(v)))

            @ui.refreshable
            def train_status() -> None:
                if vm.job:
                    with ui.row().classes("items-center gap-2"):
                        status_badge(vm.job.status)
                        ui.label(progress_text(vm.job.progress))
                if vm.losses:
                    ui.echart({
                        "xAxis": {"type": "value", "name": "iter"}, "yAxis": {"type": "value", "name": "loss"},
                        "series": [{"type": "line", "data": [list(p) for p in vm.losses], "showSymbol": False}],
                        "tooltip": {"trigger": "axis"},
                    }).classes("w-full h-56")
                if vm.job_logs:
                    log = ui.log(max_lines=300).classes("w-full h-40 text-xs")
                    log.push("\n".join(vm.job_logs[-60:]))

            train_status()
            scope.refresh_on(train_status, "job", "losses", "job_logs")

        with ui.card().classes("w-full"):
            ui.label("음악 LoRA 학습은 별도 화면에서 합니다.").classes("text-sm")
            ui.link("음악 LoRA 학습으로 이동 →", "/music-training")


@ui.page("/settings")
async def settings_page() -> None:
    vm = SettingsViewModel()
    scope = ViewScope(vm)
    with page_frame("설정·진단"):
        with section("현재 설정", "변경은 studio/.env 에서 합니다 (.env.example 참고)."):
            with ui.grid(columns=2).classes("gap-x-6 gap-y-1"):
                for k, v in vm.settings_rows():
                    ui.label(k).classes("text-gray-600")
                    ui.label(v).classes("font-mono text-sm")
        ui.button("환경 진단 실행", icon="health_and_safety", on_click=vm.run_diagnostics)

        @ui.refreshable
        def diag() -> None:
            r = vm.report
            if r is None:
                return
            with section(f"진단 결과 ({r.ok_count}/{len(r.items)} 정상)"):
                for item in r.items:
                    icon, color = ("check_circle", "positive") if item.ok else (
                        ("error", "negative") if item.severity == "error" else ("warning", "warning"))
                    with ui.row().classes("items-start gap-2 w-full"):
                        ui.icon(icon).props(f"color={color}")
                        ui.label(f"[{item.group}] {item.name}").classes("w-72")
                        with ui.column().classes("gap-0 flex-1"):
                            ui.label(item.detail).classes("text-xs font-mono break-all")
                            if item.hint:
                                ui.label(item.hint).classes("text-xs text-orange-700")

        diag()
        scope.refresh_on(diag, "report")

        @ui.refreshable
        def cap_table() -> None:
            caps = vm.capabilities
            if caps is None:
                return
            with section(f"ACE 기능 표 (출처: {caps.source})",
                         "AVAILABLE 만 사용합니다. UNBOUND 는 config/ace_routes.json 에 라우트를 등록해야 합니다."):
                rows = [{"name": c.name, "label": c.label_ko, "state": c.state.value,
                         "route": f"{c.method or ''} {c.path or ''}".strip(),
                         "fields": ", ".join(c.request_fields[:12]), "suggest": ", ".join(c.suggestions[:3]),
                         "reason": c.reason} for c in caps.capabilities.values()]
                ui.table(columns=[{"name": k, "label": lbl, "field": k, "align": "left"} for k, lbl in (
                    ("label", "기능"), ("state", "상태"), ("route", "라우트"), ("fields", "요청 필드"),
                    ("suggest", "후보 라우트"), ("reason", "비고"))], rows=rows, row_key="name").classes("w-full")
                if caps.discovered_paths:
                    with ui.expansion(f"설치본 OpenAPI 라우트 {len(caps.discovered_paths)}개").classes("w-full"):
                        ui.markdown("```\n" + "\n".join(caps.discovered_paths) + "\n```")

        cap_table()
        scope.refresh_on(cap_table, "capabilities")
    await vm.run_diagnostics()
