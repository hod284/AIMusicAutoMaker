"""곡 만들기 화면.

① AI 작사 (선택)  →  ② 가사 편집 (직접 쓰거나 AI 결과를 고침)  →  ③ 음악 프롬프트 (선택)
→  ④ 음악 생성: 음악 모델 선택 + 결과 저장 폴더 지정 → ACE-Step 으로 생성
"""

from __future__ import annotations

from nicegui import ui

from app.viewmodels.create_song_vm import CreateSongViewModel
from app.views.common import ViewScope, folder_picker, page_frame, progress_text, section, status_badge


@ui.page("/create")
def create_song_page() -> None:
    vm = CreateSongViewModel()
    scope = ViewScope(vm)
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
                        ui.button("취소", on_click=vm.cancel_job).props("flat dense color=negative")
                ui.log(max_lines=200).classes("w-full h-28 text-xs").push("\n".join(vm.job_logs[-30:]) or "…")

        job_bar()
        scope.refresh_on(job_bar, "job", "job_logs")

        with ui.row().classes("w-full gap-4 items-start no-wrap"):
            # --- ① AI 작사 (선택) ---
            with section("① AI 작사 (선택)", "AI 가 초안을 써 줍니다. 직접 쓸 거면 건너뛰고 ② 에 바로 쓰세요.").classes(
                    remove="w-full", add="w-80 shrink-0"):

                @ui.refreshable
                def lyrics_model_picker() -> None:
                    cats = {"": "전체"} | {c.id: c.name for c in vm.categories}
                    ui.select(cats, label="가사 모델 카테고리", value=vm.lyrics_filter,
                              on_change=lambda e: vm.set_lyrics_filter(e.value)).classes("w-full")
                    options = {"": "Qwen 기본 (LoRA 없음)"} | {m.id: m.label for m in vm.lyrics_models}
                    ui.select(options, label="가사 모델 (LLM)", value=vm.lyrics_model_id or "",
                              on_change=lambda e: vm.set_lyrics_model(e.value)).classes("w-full")

                lyrics_model_picker()
                scope.refresh_on(lyrics_model_picker, "categories", "lyrics_filter", "lyrics_models", "lyrics_model_id")

                form = {"topic": "", "speaker": "", "mood": "", "viewpoint": "", "forbidden": "", "variants": 1,
                        "syllable_min": 7, "syllable_max": 12,
                        "structure": "Verse 1, Pre-Chorus, Chorus, Verse 2, Chorus, Bridge, Chorus"}
                ui.input("주제 *", placeholder="비 오는 밤의 그리움").bind_value(form, "topic").classes("w-full")
                ui.input("화자").bind_value(form, "speaker").classes("w-full")
                ui.input("분위기").bind_value(form, "mood").classes("w-full")
                ui.input("시점").bind_value(form, "viewpoint").classes("w-full")
                ui.input("구조").bind_value(form, "structure").classes("w-full")
                with ui.row().classes("gap-2"):
                    ui.number("음절 최소", min=1).bind_value(form, "syllable_min").classes("w-24")
                    ui.number("음절 최대", min=1).bind_value(form, "syllable_max").classes("w-24")
                    ui.number("버전 수", min=1, max=4).bind_value(form, "variants").classes("w-20")
                ui.input("금지 표현 (쉼표)").bind_value(form, "forbidden").classes("w-full")
                ui.button("AI 가사 생성", icon="edit_note", on_click=lambda: vm.generate_lyrics(form)) \
                    .classes("w-full").mark("gen-lyrics")

                ui.separator()

                @ui.refreshable
                def drafts() -> None:
                    ui.label("최근 가사").classes("text-sm font-bold")
                    for d in vm.drafts[:15]:
                        with ui.item(on_click=lambda d=d: vm.load_draft(d.id)).classes(
                                "rounded " + ("bg-blue-50" if d.lineage_id == _lineage(vm) else "")):
                            with ui.item_section():
                                ui.item_label(d.title or "무제")
                                ui.item_label(f"v{d.version} · {d.created_at.strftime('%m-%d %H:%M')}").props("caption")

                drafts()
                scope.refresh_on(drafts, "drafts", "draft_id")

            with ui.column().classes("flex-1 min-w-0 gap-4"):
                # --- ② 가사 편집 ---
                with section("② 가사 편집", "직접 쓰거나 고치세요. 저장하면 새 버전으로 남고 이전 버전도 보존됩니다."):
                    ui.input("제목").bind_value(vm, "title").classes("w-full")
                    ui.textarea("가사", placeholder="[Verse 1]\n...\n[Chorus]\n...").bind_value(vm, "lyrics_text") \
                        .props("autogrow outlined").classes("w-full font-mono")
                    ui.button("가사 저장", icon="save", on_click=vm.save_lyrics)

                    @ui.refreshable
                    def versions() -> None:
                        if vm.history:
                            ui.label("버전: " + " → ".join(f"v{h.version}" for h in vm.history)).classes(
                                "text-xs text-gray-500")

                    versions()
                    scope.refresh_on(versions, "history")

                # --- ③ 음악 프롬프트 (선택) ---
                with section("③ 음악 프롬프트 (선택)",
                             "비워 두면 프롬프트 없이 가사와 음악 모델만으로 만듭니다. 원하면 AI 로 만들거나 직접 쓰세요."):
                    wishes = {"text": ""}
                    with ui.row().classes("w-full items-end gap-2 no-wrap"):
                        ui.input("희망 요소", placeholder="스트링 섹션, 후반 고조").bind_value(wishes, "text") \
                            .classes("flex-1")
                        ui.button("AI 프롬프트 생성", icon="auto_awesome",
                                  on_click=lambda: vm.generate_prompt(wishes["text"])).props("outline") \
                            .mark("gen-prompt")
                    ui.textarea("음악 프롬프트 (영어 권장)", placeholder="예: emotional korean ballad, piano, strings") \
                        .bind_value(vm, "prompt_text").classes("w-full")
                    ui.number("BPM (선택)", min=30, max=300).bind_value(vm, "bpm").classes("w-32")

                # --- ④ 음악 생성 ---
                with section("④ 음악 생성 (ACE-Step)", "사용할 음악 모델과 저장 폴더를 고른 뒤 만드세요."):
                    @ui.refreshable
                    def music_model_picker() -> None:
                        with ui.row().classes("w-full items-end gap-3"):
                            cats = {"": "전체"} | {c.id: c.name for c in vm.categories}
                            ui.select(cats, label="모델 카테고리", value=vm.music_filter,
                                      on_change=lambda e: vm.set_music_filter(e.value)).classes("w-48")
                            mus = {"": "ACE 기본 (LoRA 없음)"} | {m.id: m.label for m in vm.music_models}
                            ui.select(mus, label="음악 모델 *", value=vm.music_model_id or "",
                                      on_change=lambda e: vm.set_music_model(e.value)) \
                                .classes("flex-1 min-w-[260px]")
                        ui.label(vm.selection_summary).classes("text-sm text-primary")

                    music_model_picker()
                    scope.refresh_on(music_model_picker, "categories", "music_filter", "music_models",
                                     "music_model_id", "lyrics_model_id")

                    with ui.row().classes("w-full items-end gap-2 no-wrap"):
                        ui.input("결과 저장 폴더", placeholder="비우면 studio/workspace/library/songs") \
                            .bind_value(vm, "output_dir").classes("flex-1")
                        ui.button("찾아보기", icon="folder_open", on_click=lambda: folder_picker(
                            vm.browse, vm.browse_root, vm, "output_dir", title="결과 저장 폴더 선택",
                            start=vm.output_dir or None)).props("outline")

                    opts = {"seed": None, "duration": 60, "audio_format": "mp3", "ab_compare": False}
                    with ui.row().classes("items-center gap-3"):
                        ui.number("Seed (빈칸=랜덤)").bind_value(opts, "seed").classes("w-36")
                        ui.number("길이(초)", min=10, max=600).bind_value(opts, "duration").classes("w-28")
                        ui.select(["mp3", "wav", "flac"], label="포맷").bind_value(opts, "audio_format") \
                            .classes("w-24")
                        ui.checkbox("A/B 비교 (기본 vs 선택한 모델, 같은 seed)").bind_value(opts, "ab_compare")
                    ui.button("음악 생성", icon="play_circle", on_click=lambda: vm.generate_music(opts)) \
                        .mark("gen-music")

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


def _lineage(vm: CreateSongViewModel) -> str | None:
    return vm.history[0].lineage_id if vm.history else None
