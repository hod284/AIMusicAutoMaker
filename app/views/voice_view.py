"""목소리 화면 두 개: [목소리 모델 학습] (학습 / 모델·카테고리), [보컬 변환]."""

from __future__ import annotations

from fastapi.responses import RedirectResponse
from nicegui import ui

from app.viewmodels.models_vm import VoiceModelsViewModel
from app.viewmodels.voice_vm import VoiceConvertViewModel, VoiceTrainingViewModel
from app.views.common import ViewScope, folder_picker, page_frame, progress_text, section, status_badge
from app.views.models_tab import models_tab


@ui.page("/voice-models")
def voice_models_page(tab: str | None = None) -> None:
    """목소리 모델 만들기 — 음악/가사 LoRA 학습과 같은 형식: [학습] [모델 · 카테고리]."""
    vm = VoiceTrainingViewModel()
    with page_frame("목소리 모델 학습"):
        _health_banner(vm)
        with ui.tabs().classes("w-full") as tabs:
            t_train = ui.tab("학습", icon="mic")
            t_models = ui.tab("모델 · 카테고리", icon="folder_special")
        with ui.tab_panels(tabs, value=t_models if tab == "models" else t_train).classes("w-full bg-transparent"):
            with ui.tab_panel(t_train).classes("p-0 gap-4"):
                _train_tab(vm)
            with ui.tab_panel(t_models).classes("p-0"):
                models_tab(VoiceModelsViewModel())
    ui.timer(0.1, vm.check_health, once=True)   # 페이지를 먼저 보여 주고 설치 상태 확인


@ui.page("/vocal-convert")
def vocal_convert_page(song: str | None = None) -> None:   # ?song=<곡 id> 로 바로 그 곡 선택
    """보컬 변환 — 원곡 고르기/올리기 → 분리 → 목소리 변환 → 반주와 합치기 → 제목·폴더로 저장. 한 화면에서 다 한다."""
    vm = VoiceConvertViewModel(song_id=song)
    with page_frame("보컬 변환"):
        _health_banner(vm)
        _convert_tab(vm)
    ui.timer(0.1, vm.check_health, once=True)


@ui.page("/voice")
def _voice_old(song: str | None = None) -> RedirectResponse:   # 예전 주소 호환
    return RedirectResponse(f"/vocal-convert?song={song}" if song else "/voice-models")


def _health_banner(vm) -> None:  # type: ignore[no-untyped-def]
    scope = ViewScope(vm)

    @ui.refreshable
    def banner() -> None:
        h = vm.health
        if h is None or h.ok:
            if h is not None and h.is_mock:
                ui.badge("MOCK — 실제 AI 결과 아님", color="warning")
            return
        with ui.card().classes("w-full bg-red-50"):
            ui.label(f"목소리 도구가 준비되지 않았습니다: {h.message}").classes("text-negative font-bold")
            ui.label("~/AI-Music-Studio/voice-ai 에 audio-separator(.venv)와 seed-vc(seed-vc/.venv)를 설치해야 합니다. "
                     "설정 위치는 STUDIO_VOICE_ROOT 로 바꿀 수 있습니다.").classes("text-xs")

    banner()
    scope.refresh_on(banner, "health")


def _job_bar(vm, scope: ViewScope, chart: bool = False) -> None:  # type: ignore[no-untyped-def]
    @ui.refreshable
    def bar() -> None:
        job = vm.job
        if job is None:
            return
        with ui.card().classes("w-full bg-gray-50"):
            with ui.row().classes("items-center gap-3 w-full"):
                status_badge(job.status)
                ui.label(job.kind.value)
                if not job.status.is_terminal:
                    ui.spinner(size="sm")
                    ui.label(progress_text(job.progress)).classes("text-xs")
                    ui.label("실행 중에는 입력이 잠깁니다. 다른 메뉴에 다녀와도 계속 진행됩니다.") \
                        .classes("text-xs text-gray-500")
                    ui.space()
                    ui.button("중지", on_click=vm.cancel).props("flat dense color=negative")
            if chart and getattr(vm, "losses", None):
                ui.echart({
                    "xAxis": {"type": "value", "name": "step"}, "yAxis": {"type": "value", "name": "loss"},
                    "series": [{"type": "line", "data": [list(p) for p in vm.losses], "showSymbol": False}],
                    "tooltip": {"trigger": "axis"}, "grid": {"left": 50, "right": 20, "top": 30, "bottom": 30},
                }).classes("w-full h-44")
            ui.log(max_lines=300).classes("w-full h-32 text-xs").push("\n".join(vm.job_logs[-40:]) or "…")

    bar()
    scope.refresh_on(bar, "job", "job_logs", *(["losses"] if chart else []))


# ---------------------------------------------------------------------------
# 보컬 변환
# ---------------------------------------------------------------------------
def _convert_tab(vm: VoiceConvertViewModel) -> None:
    scope = ViewScope(vm)
    lockable: list[ui.element] = []

    def lock(el):  # type: ignore[no-untyped-def]
        lockable.append(el)
        return el

    def apply_lock(_v=None) -> None:  # type: ignore[no-untyped-def]
        for el in lockable:
            el.set_enabled(not vm.locked)

    _job_bar(vm, scope)
    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        # --- ① 원곡 ---
        with section("① 원곡", "보컬을 바꿀 곡. 원곡 파일은 그대로 두고 결과를 새로 저장합니다.") \
                .classes(remove="w-full", add="w-[420px] shrink-0"):
            @ui.refreshable
            def source_switch() -> None:
                ui.toggle(vm.SOURCES, value=vm.source, on_change=lambda e: vm.set_source(e.value)) \
                    .props("no-caps").set_enabled(not vm.locked)

            source_switch()
            scope.refresh_on(source_switch, "source", "job")

            song_box = ui.column().classes("w-full gap-2")
            with song_box:
                search = lock(ui.input(placeholder="곡 제목 검색", on_change=lambda e: vm.search_songs(e.value))
                              .props("dense outlined clearable").classes("w-full"))
                with search.add_slot("prepend"):
                    ui.icon("search")

                @ui.refreshable
                def song_list() -> None:
                    with ui.column().classes("w-full gap-0 max-h-[50vh] overflow-y-auto border rounded"):
                        if not vm.songs:
                            ui.label("만든 곡이 없습니다.").classes("text-xs text-gray-500 p-2")
                        for s in vm.songs:
                            item = ui.item(on_click=lambda s=s: vm.select_song(s.id)).classes(
                                "w-full border-b " + ("bg-blue-50" if s.id == vm.song_id else ""))
                            if vm.locked:
                                item.props("disable")
                            with item:
                                with ui.item_section():
                                    ui.item_label(s.title or "무제").classes("font-bold")
                                    ui.item_label(s.created_at.strftime("%Y-%m-%d %H:%M")
                                                  + (" · 보컬 변환본" if vm.is_voice_song(s) else "")).props("caption")

                song_list()
                scope.refresh_on(song_list, "songs", "song_id", "job")

            file_box = ui.column().classes("w-full gap-2")
            with file_box:
                async def on_source(e) -> None:  # type: ignore[no-untyped-def]
                    vm.upload_source(e.file.name, await e.file.read())

                lock(ui.upload(label="곡 파일 올리기 (mp3/wav/flac/ogg/m4a)", auto_upload=True, on_upload=on_source,
                               max_file_size=vm.max_upload_bytes)
                     .props("accept=.mp3,.wav,.flac,.ogg,.m4a").classes("w-full").mark("upload-source"))
                ui.label("권리가 있는 곡만 쓰세요 (직접 만든 곡, 허락받은 곡 등).").classes("text-xs text-gray-500")

            def show_source(_v=None) -> None:  # type: ignore[no-untyped-def]
                song_box.set_visibility(vm.source == "song")
                file_box.set_visibility(vm.source == "file")

            show_source()
            scope.on("source", show_source)

        with ui.column().classes("flex-1 min-w-0 gap-4"):
            @ui.refreshable
            def chosen() -> None:
                with ui.card().classes("w-full"):
                    if vm.source == "song":
                        s = vm.song
                        if s is None:
                            ui.label("왼쪽에서 곡을 고르면 여기에 나옵니다.").classes("text-gray-500")
                            return
                        ui.label(f"원곡: {s.title or '무제'}").classes("font-bold")
                        ui.audio(f"/media/songs/{s.id}").props("preload=none").classes("w-full")
                    elif vm.source_file:
                        ui.label(f"원곡 파일: {vm.label(vm.source_file)}").classes("font-bold")
                    else:
                        ui.label("왼쪽에서 곡 파일을 올리면 여기에 나옵니다.").classes("text-gray-500")

            chosen()
            scope.refresh_on(chosen, "source", "song_id", "source_file")

            # --- ② 목소리 ---
            with section("② 바꿀 목소리"):
                @ui.refreshable
                def voice_switch() -> None:
                    ui.toggle(vm.VOICES, value=vm.voice, on_change=lambda e: vm.set_voice(e.value)) \
                        .props("no-caps").set_enabled(not vm.locked)

                voice_switch()
                scope.refresh_on(voice_switch, "voice", "job")

                model_box = ui.column().classes("w-full gap-2")
                with model_box:
                    @ui.refreshable
                    def model_picker() -> None:
                        with ui.row().classes("w-full items-end gap-3"):
                            cats = {"": "전체"} | {c.id: c.name for c in vm.categories}
                            ui.select(cats, label="모델 카테고리", value=vm.model_filter,
                                      on_change=lambda e: vm.set_model_filter(e.value)).classes("w-48") \
                                .set_enabled(not vm.locked)
                            opts = {m.id: m.label + (" · 예전 참고녹음 모델" if m.train_config.get("mode") == "reference"
                                                     else "") for m in vm.models}
                            ui.select(opts, label="목소리 모델 *", value=vm.model_id,
                                      on_change=lambda e: vm.set_model(e.value)).classes("flex-1 min-w-[240px]") \
                                .set_enabled(not vm.locked)
                        if not vm.models:
                            with ui.row().classes("items-center gap-2"):
                                ui.label("목소리 모델이 없습니다.").classes("text-xs text-negative")
                                ui.link("목소리 모델 학습에서 만들기 →", "/voice-models").classes("text-xs")

                    model_picker()
                    scope.refresh_on(model_picker, "categories", "model_filter", "models", "model_id", "job")

                ref_box = ui.column().classes("w-full gap-2")
                with ref_box:
                    ui.label("바꾸고 싶은 목소리의 노래 녹음 1개(1~30초)를 올리면, 학습 없이 기본 모델이 바로 그 목소리처럼 바꿉니다. "
                             "더 닮게 하려면 [목소리 모델 학습]에서 모델을 만들어 쓰세요.") \
                        .classes("text-xs text-gray-500")

                    async def on_ref(e) -> None:  # type: ignore[no-untyped-def]
                        vm.upload_reference(e.file.name, await e.file.read())

                    lock(ui.upload(label="바꿀 목소리 녹음 올리기 (1~30초)", auto_upload=True, on_upload=on_ref,
                                   max_file_size=vm.max_upload_bytes)
                         .props("accept=.mp3,.wav,.flac,.ogg,.m4a").classes("w-full").mark("upload-reference"))

                    @ui.refreshable
                    def ref_label() -> None:
                        if vm.reference_file:
                            ui.label(f"참고 녹음: {vm.label(vm.reference_file)}").classes("text-sm font-bold")

                    ref_label()
                    scope.refresh_on(ref_label, "reference_file")
                    lock(ui.checkbox("참고 녹음에 반주가 있음 → 보컬만 뽑아서 쓰기").bind_value(vm, "reference_separate"))

                rvc_box = ui.column().classes("w-full gap-2")
                with rvc_box:
                    ui.label("다운로드한 RVC 모델(.pth)을 넣으세요. 함께 제공된 .index 파일이 있으면 추가할 수 있습니다.") \
                        .classes("text-xs text-gray-500")
                    async def on_rvc_model(e) -> None:
                        if not e.file.name.lower().endswith(".pth"):
                            vm.error = "RVC 모델은 .pth 파일을 올리세요."
                        else:
                            vm.upload_rvc(e.file.name, await e.file.read())
                        model_upload.reset()

                    async def on_rvc_index(e) -> None:
                        if not e.file.name.lower().endswith(".index"):
                            vm.error = "인덱스는 .index 파일을 올리세요."
                        else:
                            vm.upload_rvc(e.file.name, await e.file.read())
                        index_upload.reset()

                    model_upload = lock(ui.upload(label="RVC 모델 넣기 (.pth)", auto_upload=True,
                        on_upload=on_rvc_model, max_files=1, max_file_size=vm.max_upload_bytes)
                        .props("accept=.pth").classes("w-full").mark("upload-rvc-model"))
                    index_upload = lock(ui.upload(label="RVC 인덱스 넣기 (.index, 선택)", auto_upload=True,
                        on_upload=on_rvc_index, max_files=1, max_file_size=vm.max_upload_bytes)
                        .props("accept=.index").classes("w-full").mark("upload-rvc-index"))
                    @ui.refreshable
                    def rvc_files() -> None:
                        ui.label("모델: " + (vm.label(vm.rvc_model) if vm.rvc_model else "파일을 올리세요"))
                        ui.label("인덱스: " + (vm.label(vm.rvc_index) if vm.rvc_index else "없음 (선택 사항)"))
                    rvc_files()
                    scope.refresh_on(rvc_files, "rvc_model", "rvc_index")
                    ui.label("올린 파일로 바로 변환하며, 완료되면 업로드한 복사본과 입력을 비웁니다.") \
                        .classes("text-xs text-gray-500")
                    lock(ui.number("인덱스 반영 비율", min=0, max=1, step=0.1)
                         .bind_value(vm, "rvc_index_rate").classes("w-48"))

                def show_voice(_v=None) -> None:  # type: ignore[no-untyped-def]
                    model_box.set_visibility(vm.voice == "model")
                    ref_box.set_visibility(vm.voice == "reference")
                    rvc_box.set_visibility(vm.voice == "rvc")

                show_voice()
                scope.on("voice", show_voice)

            # --- ③ 변환·저장 ---
            with section("③ 변환하고 저장", "분리 → 목소리 변환 → 반주와 합치기를 한 번에 합니다."):
                with ui.row().classes("w-full items-end gap-3"):
                    lock(ui.select(vm.KEYS, label="키").bind_value(vm, "semi_tone_shift").classes("w-56")
                         .tooltip("반주 키는 그대로라 옥타브 단위만 고를 수 있습니다. 남자↔여자 목소리면 ±12"))
                    lock(ui.number("변환 스텝", min=10, max=100).bind_value(vm, "diffusion_steps").classes("w-32")
                         .tooltip("노래는 30~50 권장. 높을수록 품질↑ 시간↑")
                         .bind_visibility_from(vm, "voice", backward=lambda v: v != "rvc"))
                    lock(ui.select(["mp3", "wav", "flac"], label="저장 형식").bind_value(vm, "audio_format")
                         .classes("w-28"))
                lock(ui.input("제목", placeholder="비우면 '원곡 제목 · 목소리 이름'").bind_value(vm, "title")
                     .props("maxlength=80 counter").classes("w-full"))
                with ui.row().classes("w-full items-end gap-2 no-wrap"):
                    lock(ui.input("저장 폴더", placeholder="비우면 studio/workspace/library/songs")
                         .bind_value(vm, "output_dir").classes("flex-1"))
                    lock(ui.button("찾아보기", icon="folder_open", on_click=lambda: folder_picker(
                        vm.browse, vm.browse_root, vm, "output_dir", title="저장 폴더 선택",
                        start=vm.output_dir or None)).props("outline"))

                @ui.refreshable
                def run_button() -> None:
                    with ui.row().classes("items-center gap-3"):
                        if vm.locked:
                            ui.button("보컬 변환 중…", icon="hourglass_top").props("disable").mark("voice-convert")
                            ui.spinner("dots", size="lg", color="primary")
                            ui.label("분리 → 변환 → 합치기 순서로 진행합니다.").classes("text-xs text-gray-500")
                        else:
                            ui.button("보컬 변환", icon="record_voice_over", on_click=vm.start).mark("voice-convert")

                run_button()
                scope.refresh_on(run_button, "job")

            @ui.refreshable
            def results() -> None:
                for s in vm.results:
                    with ui.card().classes("w-full"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label(s.title).classes("font-bold")
                            if s.generation.get("is_mock"):
                                ui.badge("MOCK", color="warning")
                        ui.audio(f"/media/songs/{s.id}").classes("w-full")
                        with ui.row().classes("items-center gap-3"):
                            ui.link("다운로드", f"/media/songs/{s.id}?download=1")
                            ui.label(s.audio_path).classes("text-xs text-gray-500 break-all")
                        stems = s.generation.get("voice", {}).get("stems_dir")
                        if stems:
                            ui.label(f"바뀐 보컬·반주 따로: {stems}").classes("text-xs text-gray-400 break-all")

            results()
            scope.refresh_on(results, "results")

    apply_lock()
    scope.on("job", apply_lock)


# ---------------------------------------------------------------------------
# 목소리 학습
# ---------------------------------------------------------------------------
def _train_tab(vm: VoiceTrainingViewModel) -> None:
    scope = ViewScope(vm)
    lockable: list[ui.element] = []

    def lock(el):  # type: ignore[no-untyped-def]
        lockable.append(el)
        return el

    def apply_lock(_v=None) -> None:  # type: ignore[no-untyped-def]
        for el in lockable:
            el.set_enabled(not vm.locked)

    _job_bar(vm, scope, chart=True)

    with section("① 내 목소리 녹음 올리기",
                 "본인(또는 동의받은 사람)이 부른 노래 녹음. 반주 없이 깨끗할수록 좋습니다. 길면 15초씩 자동으로 잘라 씁니다."):
        async def on_upload(e) -> None:  # type: ignore[no-untyped-def]
            vm.upload(e.file.name, await e.file.read())

        lock(ui.upload(label="녹음 파일 선택 (여러 개 가능, mp3/wav/flac/ogg/m4a)", multiple=True, auto_upload=True,
                       on_upload=on_upload, max_file_size=vm.max_upload_bytes)
             .props("accept=.mp3,.wav,.flac,.ogg,.m4a").classes("w-full"))

        @ui.refreshable
        def staged() -> None:
            if not vm.staged:
                ui.label("올린 파일이 없습니다.").classes("text-xs text-gray-500")
                return
            with ui.row().classes("items-center gap-2"):
                ui.label(f"올린 파일 {len(vm.staged)}개").classes("text-sm font-bold")
                ui.button("모두 지우기", on_click=vm.clear_staged).props("flat dense size=sm") \
                    .set_enabled(not vm.locked)
            with ui.column().classes("w-full gap-0 max-h-48 overflow-y-auto"):
                for name in vm.staged:
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("audio_file").classes("text-gray-500")
                        ui.label(name).classes("text-sm")
                        ui.button(icon="close", on_click=lambda n=name: vm.remove_staged(n)) \
                            .props("flat dense round size=sm").set_enabled(not vm.locked)

        staged()
        scope.refresh_on(staged, "staged", "job")
        lock(ui.checkbox("녹음에 반주가 섞여 있음 → 보컬만 뽑아서 쓰기 (audio-separator)").bind_value(vm, "separate"))

    with section("② 목소리 모델 만들기"):
        with ui.row().classes("w-full items-end gap-3"):
            lock(ui.input("목소리 모델 이름 *", placeholder="예: 내 목소리 허스키").bind_value(vm, "model_name")
                 .props("maxlength=80 counter").classes("w-72"))
            lock(ui.input("결과 저장 폴더", placeholder="비우면 studio/workspace/library/voice/models")
                 .bind_value(vm, "export_dir").classes("flex-1 min-w-[260px]"))
            lock(ui.button("찾아보기", icon="folder_open", on_click=lambda: folder_picker(
                vm.browse, vm.browse_root, vm, "export_dir", title="결과 저장 폴더 선택",
                start=vm.export_dir or None)).props("outline"))
        @ui.refreshable
        def train_opts() -> None:
            with ui.row().classes("items-end gap-3"):
                ui.number("학습 스텝", min=50, max=20000, step=50).bind_value(vm, "max_steps").classes("w-36") \
                    .set_enabled(not vm.locked)
                ui.number("배치 크기", min=1, max=16).bind_value(vm, "batch_size").classes("w-28") \
                    .set_enabled(not vm.locked)
                ui.label("처음엔 500 스텝으로 시험해 보세요. 녹음이 많을수록 스텝을 늘리면 좋습니다.") \
                    .classes("text-xs text-gray-500")

        train_opts()
        scope.refresh_on(train_opts, "job")

        @ui.refreshable
        def start_button() -> None:
            with ui.row().classes("items-center gap-3"):
                if vm.locked:
                    ui.button("만드는 중…", icon="hourglass_top").props("disable").mark("voice-train")
                    ui.spinner("dots", size="lg", color="primary")
                else:
                    ui.button("목소리 모델 만들기", icon="mic", on_click=vm.start).mark("voice-train")

        start_button()
        scope.refresh_on(start_button, "job")
        ui.label("만든 모델은 [모델 · 카테고리] 탭에서 카테고리로 묶고, [보컬 변환] 메뉴에서 씁니다.") \
            .classes("text-xs text-gray-500")

    apply_lock()
    scope.on("job", apply_lock)
