"""곡 만들기 화면.

① 가사 선택 (가사집에서 고르기만 — 쓰기·고치기는 가사집)  →  ② 음악 프롬프트 (선택)
→  ③ 음악 생성: 음악 모델·노래 언어·저장 폴더 → ACE-Step 으로 생성
"""

from __future__ import annotations

from nicegui import ui

from app.viewmodels.create_song_vm import CreateSongViewModel
from app.views.common import ViewScope, folder_picker, page_frame, progress_text, section, status_badge


@ui.page("/create")
def create_song_page(lyric: str | None = None) -> None:   # ?lyric=<가사집 가사 id>
    vm = CreateSongViewModel(lyric_id=lyric)
    client = ui.context.client
    vm.is_live = lambda: client.has_socket_connection and not client.is_deleted
    scope = ViewScope(vm)
    lockable: list[ui.element] = []      # 생성 중에 잠글 입력들

    def lock(el):  # type: ignore[no-untyped-def]
        lockable.append(el)
        return el

    def apply_lock(_v=None) -> None:  # type: ignore[no-untyped-def]
        for el in lockable:
            el.set_enabled(not vm.locked)

    with page_frame("곡 만들기"):
        # --- 작업 상태 바 ---
        @ui.refreshable
        def job_bar() -> None:
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
                        ui.label("만드는 동안에는 화면을 바꿀 수 없습니다. 다른 메뉴에 다녀와도 계속 진행됩니다.") \
                            .classes("text-xs text-gray-500")
                        ui.space()
                        ui.button("취소", on_click=vm.cancel_job).props("flat dense color=negative")
                ui.log(max_lines=200).classes("w-full h-28 text-xs").push("\n".join(vm.job_logs[-30:]) or "…")

        job_bar()
        scope.refresh_on(job_bar, "job", "job_logs")

        with ui.row().classes("w-full gap-4 items-start no-wrap"):
            # --- ① 가사 선택 (가사집) ---
            with section("① 가사 선택", "가사집에서 만든 가사 중 하나를 고르세요.").classes(
                    remove="w-full", add="w-80 shrink-0"):

                @ui.refreshable
                def lyric_filter() -> None:
                    options = {"": "전체"} | {c.id: f"{c.name} ({c.count})" for c in vm.lyric_categories}
                    ui.select(options, label="가사 카테고리", value=vm.lyric_filter,
                              on_change=lambda e: vm.set_lyric_filter(e.value)).classes("w-full") \
                        .set_enabled(not vm.locked)

                lyric_filter()
                scope.refresh_on(lyric_filter, "lyric_categories", "lyric_filter", "job")

                draft_search = lock(ui.input(placeholder="제목·가사 내용 검색", value=vm.draft_query,
                                             on_change=lambda e: vm.search_drafts(e.value))
                                    .props("dense outlined clearable").classes("w-full"))
                with draft_search.add_slot("prepend"):
                    ui.icon("search")

                @ui.refreshable
                def drafts() -> None:
                    ui.label(f"{len(vm.drafts)} / {vm.draft_total}개").classes("text-xs text-gray-500")
                    with ui.column().classes("w-full gap-0 max-h-[55vh] overflow-y-auto border rounded"):
                        if not vm.drafts:
                            ui.label("조건에 맞는 가사가 없습니다." if (vm.draft_query or vm.lyric_filter)
                                     else "가사집에 가사가 없습니다.").classes("text-xs text-gray-500 p-2")
                        for d in vm.drafts:
                            item = ui.item(on_click=lambda d=d: vm.select_lyric(d.id)).classes(
                                "w-full border-b " + ("bg-blue-50" if d.id == vm.draft_id else ""))
                            if vm.locked:
                                item.props("disable")
                            with item:
                                with ui.item_section():
                                    ui.item_label(d.title or "무제").classes("font-bold")
                                    ui.item_label((d.updated_at or d.created_at).strftime("%Y-%m-%d %H:%M")) \
                                        .props("caption")
                        if len(vm.drafts) < vm.draft_total:
                            ui.button(f"더 보기 (남은 {vm.draft_total - len(vm.drafts)}개)", on_click=vm.more_drafts) \
                                .props("flat dense").classes("w-full")

                drafts()
                scope.refresh_on(drafts, "drafts", "draft_id", "draft_total", "job")
                ui.link("가사집에서 새로 만들기 / 고치기 →", "/lyrics").classes("text-xs")

            with ui.column().classes("flex-1 min-w-0 gap-4"):
                # --- 고른 가사 (보기 전용) ---
                @ui.refreshable
                def chosen() -> None:
                    with ui.card().classes("w-full"):
                        if not vm.draft_id:
                            with ui.column().classes("w-full items-center py-8"):
                                ui.icon("library_books").classes("text-5xl text-gray-300")
                                ui.label("왼쪽에서 가사를 고르면 여기에 나옵니다.").classes("text-gray-500")
                            return
                        with ui.row().classes("w-full items-center"):
                            ui.label(vm.title or "무제").classes("text-base font-bold")
                            ui.space()
                            ui.button("가사집에서 고치기", icon="edit",
                                      on_click=lambda: ui.navigate.to(f"/lyrics?open={vm.draft_id}")) \
                                .props("flat dense").set_enabled(not vm.locked)
                        with ui.scroll_area().classes("w-full h-[45vh] border rounded bg-gray-50"):
                            ui.label(vm.lyrics_text).classes("whitespace-pre-wrap font-mono text-sm p-3")

                chosen()
                scope.refresh_on(chosen, "draft_id", "title", "lyrics_text", "job")

                # --- ② 음악 프롬프트 (선택) ---
                with section("② 음악 프롬프트 (선택)",
                             "비워 두면 프롬프트 없이 가사와 음악 모델만으로 만듭니다. 원하면 AI 로 만들거나 직접 쓰세요."):
                    with ui.row().classes("w-full items-end gap-2 no-wrap"):
                        lock(ui.input("희망 요소", placeholder="스트링 섹션, 후반 고조").bind_value(vm, "wishes")
                             .classes("flex-1"))
                        lock(ui.button("AI 프롬프트 생성", icon="auto_awesome",
                                       on_click=lambda: vm.generate_prompt()).props("outline")
                             .mark("gen-prompt"))
                    lock(ui.textarea("음악 프롬프트 (영어 권장)", placeholder="예: emotional korean ballad, piano, strings")
                         .bind_value(vm, "prompt_text").classes("w-full"))
                    lock(ui.number("BPM (선택)", min=30, max=300).bind_value(vm, "bpm").classes("w-32"))

                # --- ③ 음악 생성 ---
                with section("③ 음악 생성 (ACE-Step)", "음악 모델·노래 언어·저장 폴더를 고른 뒤 만드세요."):
                    @ui.refreshable
                    def music_model_picker() -> None:
                        with ui.row().classes("w-full items-end gap-3"):
                            cats = {"": "전체"} | {c.id: c.name for c in vm.categories}
                            ui.select(cats, label="모델 카테고리", value=vm.music_filter,
                                      on_change=lambda e: vm.set_music_filter(e.value)).classes("w-48") \
                                .set_enabled(not vm.locked)
                            mus = {"": "ACE 기본 (LoRA 없음)"} | {m.id: m.label for m in vm.music_models}
                            ui.select(mus, label="음악 모델 *", value=vm.music_model_id or "",
                                      on_change=lambda e: vm.set_music_model(e.value)) \
                                .classes("flex-1 min-w-[260px]").set_enabled(not vm.locked)
                        ui.label(vm.selection_summary).classes("text-sm text-primary")

                    music_model_picker()
                    scope.refresh_on(music_model_picker, "categories", "music_filter", "music_models",
                                     "music_model_id", "title", "language", "job")

                    with ui.row().classes("w-full items-end gap-3 no-wrap"):
                        lock(ui.select(vm.LANGUAGES, label="노래 언어").bind_value(vm, "language").classes("w-56")
                             .tooltip("가사를 고르면 가사 글자를 보고 자동으로 정해집니다. 바꿀 수 있습니다."))
                        lock(ui.input("곡 제목", placeholder="비우면 고른 가사의 제목을 씁니다")
                             .bind_value(vm, "song_title").props("maxlength=80").classes("flex-1"))
                    with ui.row().classes("w-full items-end gap-2 no-wrap"):
                        lock(ui.input("결과 저장 폴더", placeholder="비우면 studio/workspace/library/songs")
                             .bind_value(vm, "output_dir").classes("flex-1"))
                        lock(ui.button("찾아보기", icon="folder_open", on_click=lambda: folder_picker(
                            vm.browse, vm.browse_root, vm, "output_dir", title="결과 저장 폴더 선택",
                            start=vm.output_dir or None)).props("outline"))

                    with ui.row().classes("items-center gap-3"):
                        lock(ui.number("Seed (빈칸=랜덤)").bind_value(vm, "seed").classes("w-36"))
                        lock(ui.number("길이(초)", min=10, max=600).bind_value(vm, "duration").classes("w-28"))
                        lock(ui.select(["mp3", "wav", "flac"], label="포맷").bind_value(vm, "audio_format")
                             .classes("w-24"))
                        lock(ui.checkbox("A/B 비교 (기본 vs 선택한 모델, 같은 seed)").bind_value(vm, "ab_compare"))

                    @ui.refreshable
                    def generate_button() -> None:
                        with ui.row().classes("items-center gap-3"):
                            if vm.locked:
                                ui.button("음악 생성 중…", icon="hourglass_top").props("disable").mark("gen-music")
                                ui.spinner("dots", size="lg", color="primary")
                                ui.label("완성되면 이 화면이 처음 상태로 돌아가고, 결과가 아래에 나옵니다.") \
                                    .classes("text-xs text-gray-500")
                            else:
                                ui.button("음악 생성", icon="play_circle", on_click=lambda: vm.generate_music()) \
                                    .mark("gen-music")

                    generate_button()
                    scope.refresh_on(generate_button, "job")

                    @ui.refreshable
                    def results() -> None:
                        for song in vm.songs:
                            with ui.card().classes("w-full"):
                                with ui.row().classes("items-center gap-2"):
                                    ui.label(song.title).classes("font-bold")
                                    if song.generation.get("is_mock"):
                                        ui.badge("MOCK", color="warning")
                                    ui.label(f"seed {song.generation.get('seed', '-')}").classes("text-xs")
                                ui.audio(f"/media/songs/{song.id}").classes("w-full")
                                with ui.row().classes("items-center gap-3"):
                                    ui.link("다운로드", f"/media/songs/{song.id}?download=1")
                                    ui.label(song.audio_path).classes("text-xs text-gray-500 break-all")

                    results()
                    scope.refresh_on(results, "songs")

        apply_lock()
        scope.on("job", apply_lock)

