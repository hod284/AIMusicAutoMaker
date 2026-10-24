"""Library / Jobs / Training / Settings 페이지."""

from __future__ import annotations

from nicegui import ui

from app.adapters.ports import CapabilityState
from app.viewmodels.misc_vms import JobsViewModel, LibraryViewModel, SettingsViewModel, TrainingViewModel
from app.viewmodels.models_vm import LyricsModelsViewModel
from app.views.common import ViewScope, folder_picker, page_frame, progress_text, section, status_badge
from app.views.models_tab import models_tab

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
        with ui.tabs().classes("w-full") as tabs:
            t_train = ui.tab("학습", icon="model_training")
            t_models = ui.tab("모델 · 카테고리", icon="folder_special")
        with ui.tab_panels(tabs, value=t_train).classes("w-full bg-transparent"):
            with ui.tab_panel(t_train).classes("p-0 gap-4"):
                _lyrics_train_tab(vm, scope)
            with ui.tab_panel(t_models).classes("p-0"):
                models_tab(LyricsModelsViewModel())


def _lyrics_train_tab(vm: TrainingViewModel, scope: ViewScope) -> None:
    # --- 결과 이름 / 저장 폴더 ---
    with ui.row().classes("w-full items-center gap-4"):
        ui.input("결과 모델 이름 *", placeholder="예: 감성 발라드 작사 v1").bind_value(vm, "model_title").props(
            "maxlength=80 counter").classes("w-80")
        ui.input("결과 저장 폴더", placeholder="비우면 studio/workspace/library/lyrics/adapters") \
            .bind_value(vm, "export_dir").classes("flex-1 min-w-[280px]")
        ui.button("찾아보기", icon="folder_open", on_click=lambda: folder_picker(
            vm.browse, vm.browse_root, vm, "export_dir", title="결과 저장 폴더 선택",
            start=vm.export_dir or None)).props("outline")
    ui.label("결과는 '결과 저장 폴더/결과 모델 이름' 폴더에 저장됩니다. 같은 이름 폴더가 있으면 덮어쓰지 않고 멈춥니다.") \
        .classes("text-xs text-gray-500")

    # --- ① 데이터셋 ---
    with section("① 데이터셋 만들기 — 가사 .txt 올리기",
                 "가사 1곡 = .txt 파일 1개. 여러 개를 한 번에 올릴 수 있습니다. 권리가 확인된 가사만 사용하세요."):
        ui.label("파일 안에는 가사만 적으면 됩니다. [Verse 1], [Chorus] 같은 구조 표시를 넣으면 더 좋습니다. "
                 "맨 위에 '제목: …', '주제: …', '분위기: …' 줄을 넣으면 그대로 쓰고, 없으면 파일 이름을 제목/주제로 씁니다.") \
            .classes("text-xs text-gray-500")

        async def on_txt(e) -> None:  # type: ignore[no-untyped-def]
            vm.add_txt(e.file.name, await e.file.read())

        uploader = ui.upload(label="가사 .txt 올리기 (여러 개 선택 가능)", multiple=True, auto_upload=True,
                             on_upload=on_txt, max_file_size=200_000).props("accept=.txt").classes("w-full")

        @ui.refreshable
        def txt_list() -> None:
            files = vm.txt_files
            with ui.row().classes("items-center gap-3"):
                ui.label(f"올린 가사 {len(files)}곡").classes("text-sm font-bold")
                if files:
                    ui.button("모두 빼기", on_click=lambda: (vm.clear_txt(), uploader.reset())) \
                        .props("flat dense color=negative size=sm")
            with ui.column().classes("w-full gap-1 max-h-72 overflow-y-auto"):
                for name, text in files:
                    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
                    with ui.row().classes("w-full items-center gap-2 no-wrap border rounded px-2 py-1"):
                        ui.icon("description", color="grey")
                        ui.label(name).classes("font-mono text-sm w-56 ellipsis")
                        ui.label(first[:60]).classes("text-xs text-gray-500 flex-1 ellipsis")
                        ui.button(icon="visibility", on_click=lambda n=name: _show_preview(vm, n)) \
                            .props("flat dense round").tooltip("학습 예제로 어떻게 바뀌는지 보기")
                        ui.button(icon="close", on_click=lambda n=name: vm.remove_txt(n)) \
                            .props("flat dense round color=negative")
            ui.button("이 가사들로 데이터셋 버전 만들기", icon="create_new_folder", on_click=vm.build_from_txt) \
                .set_enabled(len(files) >= 2)

        txt_list()
        scope.refresh_on(txt_list, "txt_files")

        @ui.refreshable
        def report() -> None:
            r = vm.jsonl_report
            if r:
                color = "text-negative" if r.errors else "text-positive"
                ui.label(f"전체 {r.total} · 유효 {r.valid} · 중복 {r.duplicates} · 오류 {len(r.errors)}").classes(
                    f"text-sm font-bold {color}")
                for msg in r.errors[:10] + r.warnings:
                    ui.label(msg).classes("text-xs text-orange-700")

        report()
        scope.refresh_on(report, "jsonl_report")

        with ui.expansion("고급: JSONL 로 직접 만들기", icon="code").classes("w-full"):
            text = {"value": ""}

            async def on_upload(e) -> None:  # type: ignore[no-untyped-def]
                text["value"] = await e.file.text()
                vm.preview_jsonl(text["value"])

            ui.upload(label="JSONL 업로드", on_upload=on_upload, auto_upload=True, max_file_size=20_000_000).props(
                "accept=.jsonl").classes("w-full")
            ui.textarea("또는 붙여넣기 (한 줄에 {\"messages\":[user, assistant]})").bind_value(text, "value") \
                .props("outlined").classes("w-full font-mono text-xs")
            with ui.row().classes("gap-2"):
                ui.button("검증", icon="fact_check", on_click=lambda: vm.preview_jsonl(text["value"])).props("outline")
                ui.button("JSONL 로 데이터셋 버전 만들기", on_click=lambda: vm.build_dataset(text["value"])) \
                    .props("outline")

    # --- ② 학습 ---
    with section("② 학습 (mlx_lm.lora)", "비운 값은 mlx_lm 기본값을 씁니다. 학습 중에는 가사 생성이 잠시 멈출 수 있습니다."):
        @ui.refreshable
        def dataset_picker() -> None:
            if not vm.datasets:
                ui.label("데이터셋이 없습니다. ① 에서 먼저 만드세요.").classes("text-sm text-gray-500")
                return
            options = {d.id: f"{vm.dataset_label(d)} · {vm.dataset_counts(d)} · {d.created_at.strftime('%m-%d %H:%M')}"
                       for d in reversed(vm.datasets)}
            ui.select(options, label="데이터셋 *", value=vm.dataset_id,
                      on_change=lambda e: vm.select_dataset(e.value)).classes("w-full")
            if vm.dataset:
                ui.label(vm.dataset.source_path).classes("text-xs text-gray-400 break-all")

        dataset_picker()
        scope.refresh_on(dataset_picker, "datasets", "dataset_id")

        form = {"iters": 200, "batch_size": 1, "learning_rate": None, "num_layers": None,
                "max_seq_length": None, "save_every": None}
        with ui.row().classes("items-end gap-3"):
            ui.number("반복 횟수 (iters)", min=1).bind_value(form, "iters").classes("w-36")
            ui.number("batch size", min=1).bind_value(form, "batch_size").classes("w-28")
            ui.number("learning rate", format="%.6f", placeholder="기본").bind_value(form, "learning_rate") \
                .classes("w-36")
            ui.number("LoRA 레이어 수", min=1, placeholder="기본").bind_value(form, "num_layers").classes("w-32")
            ui.number("최대 길이 (토큰)", min=128, placeholder="기본").bind_value(form, "max_seq_length") \
                .classes("w-36")
            ui.number("저장 주기 (iter)", min=1, placeholder="기본").bind_value(form, "save_every").classes("w-32")

        @ui.refreshable
        def buttons() -> None:
            with ui.row().classes("items-center gap-2"):
                start = ui.button("학습 시작", icon="play_arrow", on_click=lambda: vm.start_training(form))
                start.set_enabled(not vm.training)
                if vm.training:
                    ui.button("중지", icon="stop", on_click=vm.cancel).props("outline color=negative")

        buttons()
        scope.refresh_on(buttons, "job")

        @ui.refreshable
        def train_status() -> None:
            job = vm.job
            if job is None:
                return
            with ui.card().classes("w-full bg-gray-50"):
                with ui.row().classes("items-center gap-3 w-full"):
                    status_badge(job.status)
                    if not job.status.is_terminal:
                        ui.spinner(size="sm")
                    ui.label(progress_text(job.progress)).classes("text-sm")
                    if vm.losses:
                        ui.label(f"iter {vm.losses[-1][0]} · train loss {vm.losses[-1][1]:.3f}").classes("text-sm")
                    if vm.val_losses:
                        ui.label(f"val loss {vm.val_losses[-1][1]:.3f}").classes("text-sm text-gray-600")
                if job.progress is not None:
                    ui.linear_progress(value=job.progress, show_value=False).classes("w-full")
                if job.error:
                    ui.label(job.error).classes("text-sm text-negative break-all")
            if vm.losses or vm.val_losses:
                ui.echart({
                    "legend": {"data": ["train", "val"]},
                    "xAxis": {"type": "value", "name": "iter"}, "yAxis": {"type": "value", "name": "loss"},
                    "series": [
                        {"name": "train", "type": "line", "data": [list(p) for p in vm.losses], "showSymbol": False},
                        {"name": "val", "type": "line", "data": [list(p) for p in vm.val_losses]},
                    ],
                    "tooltip": {"trigger": "axis"},
                }).classes("w-full h-56")
            if vm.job_logs:
                log = ui.log(max_lines=300).classes("w-full h-40 text-xs")
                log.push("\n".join(vm.job_logs[-80:]))

        train_status()
        scope.refresh_on(train_status, "job", "losses", "val_losses", "job_logs")


def _show_preview(vm: TrainingViewModel, filename: str) -> None:
    with ui.dialog() as dialog, ui.card().classes("w-[640px] max-w-full"):
        ui.label(f"{filename} → 학습 예제").classes("font-bold")
        ui.label("[요청] 은 곡 만들기 화면이 보내는 형식, [정답 가사] 는 모델이 배울 결과입니다.").classes(
            "text-xs text-gray-500")
        ui.code(vm.preview_txt(filename), language="text").classes("w-full max-h-[60vh] overflow-y-auto")
        ui.button("닫기", on_click=dialog.close).props("flat")
    dialog.open()


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
