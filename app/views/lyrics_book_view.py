"""가사집 화면.

[가사 목록] 카테고리 필터(기본 '전체')·검색 → 목록에서 가사를 누르면 오른쪽 에디터에서 고치고 [저장] (덮어쓰기)
            [새 가사] / [AI로 가사 만들기] 로 새 가사를 만든다.
[카테고리 편집] 카테고리 만들기·삭제, 가사 넣기·빼기, 가사 삭제
"""

from __future__ import annotations

from nicegui import ui

from app.domain.enums import JobStatus
from app.viewmodels.lyrics_book_vm import LyricCategoriesViewModel, LyricsBookViewModel
from app.views.common import ViewScope, page_frame, progress_text, status_badge


@ui.page("/lyrics")
def lyrics_book_page(open: str | None = None) -> None:  # noqa: A002 - ?open=<가사 id>
    with page_frame("가사집"):
        with ui.tabs().classes("w-full") as tabs:
            t_list = ui.tab("가사 목록", icon="library_books")
            t_edit = ui.tab("카테고리 편집", icon="folder_special")
        with ui.tab_panels(tabs, value=t_list).classes("w-full bg-transparent"):
            with ui.tab_panel(t_list).classes("p-0"):
                _list_tab(LyricsBookViewModel(open_id=open))
            with ui.tab_panel(t_edit).classes("p-0"):
                _category_tab(LyricCategoriesViewModel())


# ---------------------------------------------------------------------------
# [가사 목록] 탭
# ---------------------------------------------------------------------------
def _list_tab(vm: LyricsBookViewModel) -> None:
    scope = ViewScope(vm)
    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        # --- 왼쪽: 필터·검색·목록 ---
        with ui.card().classes("w-96 shrink-0 gap-2"):
            with ui.row().classes("w-full gap-2"):
                ui.button("새 가사", icon="add", on_click=vm.new).props("dense").classes("flex-1")
                ui.button("AI로 가사 만들기", icon="auto_awesome", on_click=lambda: _ai_dialog(vm)) \
                    .props("dense outline").classes("flex-1")

            @ui.refreshable
            def filter_select() -> None:
                options = {"": "전체"} | {c.id: f"{c.name} ({c.count})" for c in vm.categories}
                ui.select(options, label="카테고리", value=vm.category_filter,
                          on_change=lambda e: vm.set_filter(e.value)).classes("w-full")

            filter_select()
            scope.refresh_on(filter_select, "categories", "category_filter")

            search = ui.input(placeholder="제목·가사 내용 검색", on_change=lambda e: vm.set_query(e.value)) \
                .props("dense outlined clearable").classes("w-full")
            with search.add_slot("prepend"):
                ui.icon("search")

            @ui.refreshable
            def listing() -> None:
                ui.label(f"{len(vm.items)} / {vm.total}개").classes("text-xs text-gray-500")
                with ui.column().classes("w-full gap-0 max-h-[60vh] overflow-y-auto border rounded"):
                    if not vm.items:
                        ui.label("조건에 맞는 가사가 없습니다." if (vm.query or vm.category_filter)
                                 else "아직 가사가 없습니다. [새 가사] 나 [AI로 가사 만들기] 로 시작하세요.") \
                            .classes("text-xs text-gray-500 p-2")
                    for d in vm.items:
                        active = d.id == vm.selected_id
                        with ui.item(on_click=lambda d=d: vm.open(d.id)).classes(
                                "w-full border-b " + ("bg-blue-50" if active else "")):
                            with ui.item_section():
                                ui.item_label(d.title or "무제").classes("font-bold")
                                first = next((ln for ln in d.lyrics.splitlines()
                                              if ln.strip() and not ln.strip().startswith("[")), "")
                                ui.item_label(first[:40]).props("caption")
                                with ui.row().classes("gap-1 items-center"):
                                    when = d.updated_at or d.created_at
                                    ui.label(when.strftime("%Y-%m-%d %H:%M")).classes("text-xs text-gray-400")
                                    for name in vm.category_names(d.id):
                                        ui.badge(name, color="primary").props("outline")
                    if len(vm.items) < vm.total:
                        ui.button(f"더 보기 (남은 {vm.total - len(vm.items)}개)", on_click=vm.more) \
                            .props("flat dense").classes("w-full")

            listing()
            scope.refresh_on(listing, "items", "total", "selected_id", "membership")

        # --- 오른쪽: AI 작업 상태 + 에디터 ---
        with ui.column().classes("flex-1 min-w-0 gap-3"):
            @ui.refreshable
            def job_bar() -> None:
                job = vm.job
                if job is None or job.status == JobStatus.SUCCEEDED or job.status == JobStatus.CANCELLED:
                    return
                with ui.card().classes("w-full bg-gray-50"):
                    with ui.row().classes("items-center gap-3 w-full"):
                        status_badge(job.status)
                        ui.label("AI 작사")
                        if not job.status.is_terminal:
                            ui.spinner(size="sm")
                            ui.label(progress_text(job.progress)).classes("text-xs")
                            ui.button("취소", on_click=vm.cancel_job).props("flat dense color=negative")
                    if vm.job_logs:
                        ui.label(vm.job_logs[-1]).classes("text-xs text-gray-500 break-all")

            job_bar()
            scope.refresh_on(job_bar, "job", "job_logs")

            @ui.refreshable
            def editor() -> None:
                if not vm.editor_open:
                    with ui.card().classes("w-full items-center justify-center h-64"):
                        ui.icon("edit_note").classes("text-5xl text-gray-300")
                        ui.label("왼쪽 목록에서 가사를 고르거나 [새 가사] 를 누르세요.").classes("text-gray-500")
                    return
                with ui.card().classes("w-full gap-3"):
                    with ui.row().classes("w-full items-center"):
                        ui.label("새 가사" if vm.selected_id is None else "가사 편집").classes("text-base font-bold")
                        ui.space()
                        ui.button(icon="close", on_click=vm.close_editor).props("flat round dense") \
                            .tooltip("에디터 닫기")
                    ui.input("제목", placeholder="비우면 첫 가사 줄로 정해집니다").bind_value(vm, "title") \
                        .props("maxlength=80").classes("w-full")
                    cats = {c.id: c.name for c in vm.categories}
                    ui.select(cats, label="카테고리 (여러 개 선택 가능)", multiple=True,
                              value=[c for c in vm.selected_categories if c in cats],
                              on_change=lambda e: setattr(vm, "selected_categories", list(e.value or []))) \
                        .props("use-chips").classes("w-full")
                    if not cats:
                        ui.label("카테고리는 [카테고리 편집] 탭에서 만들 수 있습니다.").classes("text-xs text-gray-500")
                    ui.textarea("가사", placeholder="[Verse 1]\n...\n\n[Chorus]\n...").bind_value(vm, "text") \
                        .props("outlined rows=22").classes("w-full font-mono")
                    with ui.row().classes("w-full items-center gap-2"):
                        ui.button("저장", icon="save", on_click=vm.save)
                        if vm.selected_id:
                            ui.button("이 가사로 곡 만들기", icon="music_note",
                                      on_click=lambda: ui.navigate.to(f"/create?lyric={vm.selected_id}")) \
                                .props("outline")
                        ui.space()
                        ui.button("삭제", icon="delete", on_click=lambda: _confirm_delete(vm)) \
                            .props("flat color=negative")
                    ui.label("저장하면 이 가사에 그대로 덮어씁니다 (버전을 따로 만들지 않음).") \
                        .classes("text-xs text-gray-500")

            editor()
            scope.refresh_on(editor, "editor_open", "selected_id", "categories")


def _ai_dialog(vm: LyricsBookViewModel) -> None:
    form = {"topic": "", "speaker": "", "mood": "", "viewpoint": "",
            "structure": "Verse 1, Pre-Chorus, Chorus, Verse 2, Chorus, Bridge, Chorus"}
    with ui.dialog() as dialog, ui.card().classes("w-[560px] max-w-full gap-2"):
        ui.label("AI로 가사 만들기").classes("text-base font-bold")
        ui.label("만든 가사는 가사집에 바로 저장되고 에디터에 열립니다.").classes("text-xs text-gray-500")
        ui.select(vm.LANGUAGES, label="가사 언어 (노래 언어)").bind_value(vm, "language").classes("w-full")
        models = {"": "Qwen 기본 (LoRA 없음)"} | {m.id: m.label for m in vm.lyrics_models}
        ui.select(models, label="가사 모델", value=vm.lyrics_model_id or "",
                  on_change=lambda e: setattr(vm, "lyrics_model_id", e.value or None)).classes("w-full")
        ui.input("주제 *", placeholder="비 오는 밤의 그리움").bind_value(form, "topic").classes("w-full")
        with ui.row().classes("w-full gap-2 no-wrap"):
            ui.input("화자").bind_value(form, "speaker").classes("flex-1")
            ui.input("분위기").bind_value(form, "mood").classes("flex-1")
            ui.input("시점").bind_value(form, "viewpoint").classes("flex-1")
        ui.input("구조").bind_value(form, "structure").classes("w-full")
        with ui.row().classes("w-full justify-end"):
            ui.button("취소", on_click=dialog.close).props("flat")
            ui.button("만들기", icon="auto_awesome",
                      on_click=lambda: (vm.generate(form), dialog.close() if not vm.error else None))
    dialog.open()


def _confirm_delete(vm: LyricsBookViewModel) -> None:
    if vm.selected_id is None:
        vm.close_editor()
        return
    with ui.dialog() as dialog, ui.card():
        ui.label(f"'{vm.title or '무제'}' 가사를 삭제할까요?").classes("font-bold")
        ui.label("가사집과 모든 카테고리에서 지웁니다. 이 가사로 만든 곡은 남습니다.").classes("text-sm")
        with ui.row().classes("w-full justify-end"):
            ui.button("취소", on_click=dialog.close).props("flat")
            ui.button("삭제", on_click=lambda: (vm.delete(), dialog.close())).props("color=negative")
    dialog.open()


# ---------------------------------------------------------------------------
# [카테고리 편집] 탭
# ---------------------------------------------------------------------------
def _category_tab(vm: LyricCategoriesViewModel) -> None:
    scope = ViewScope(vm)
    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        with ui.card().classes("w-[440px] shrink-0"):
            ui.label("카테고리").classes("font-bold")
            new_cat = {"name": ""}
            with ui.row().classes("w-full items-end gap-2"):
                ui.input("새 카테고리 이름").bind_value(new_cat, "name").props("maxlength=40").classes("flex-1")
                ui.button("만들기", icon="add",
                          on_click=lambda: (vm.create(new_cat["name"]), new_cat.update(name=""))).props("outline dense")

            @ui.refreshable
            def category_panel() -> None:
                if not vm.categories:
                    ui.label("카테고리가 없습니다. 위에서 만드세요.").classes("text-xs text-gray-500")
                    return
                with ui.row().classes("w-full items-end gap-2 no-wrap"):
                    ui.select({c.id: f"{c.name} ({c.count})" for c in vm.categories}, label="편집할 카테고리",
                              value=vm.category_id, on_change=lambda e: vm.select(e.value)).classes("flex-1")
                    ui.button(icon="delete", on_click=lambda: _confirm_delete_category(vm)) \
                        .props("flat round color=negative").tooltip("카테고리 삭제 (가사는 남음)")
                ui.label(f"이 카테고리의 가사 {len(vm.members)}개").classes("text-sm font-bold mt-2")
                with ui.column().classes("w-full gap-1 max-h-[60vh] overflow-y-auto"):
                    if not vm.members:
                        ui.label("오른쪽 목록에서 [넣기] 로 가사를 추가하세요.").classes("text-xs text-gray-500")
                    for d in vm.members:
                        with ui.row().classes("w-full items-center gap-2 no-wrap border rounded px-2 py-1"):
                            ui.label(d.title or "무제").classes("flex-1 ellipsis")
                            ui.button(icon="remove_circle_outline", on_click=lambda d=d: vm.remove(d.id)) \
                                .props("flat dense round color=negative").tooltip("카테고리에서 빼기")

            category_panel()
            scope.refresh_on(category_panel, "categories", "category_id", "members")

        with ui.card().classes("flex-1 min-w-0"):
            ui.label("가사 전체").classes("font-bold")
            search = ui.input(placeholder="제목·가사 내용 검색", on_change=lambda e: vm.set_query(e.value)) \
                .props("dense outlined clearable").classes("w-full")
            with search.add_slot("prepend"):
                ui.icon("search")

            @ui.refreshable
            def all_list() -> None:
                ui.label(f"{len(vm.all_items)} / {vm.total}개").classes("text-xs text-gray-500")
                with ui.column().classes("w-full gap-1 max-h-[65vh] overflow-y-auto"):
                    if not vm.all_items:
                        ui.label("가사가 없습니다.").classes("text-xs text-gray-500")
                    for d in vm.all_items:
                        inside = d.id in vm.member_ids
                        with ui.row().classes("w-full items-center gap-2 no-wrap border rounded px-2 py-1"):
                            with ui.column().classes("gap-0 flex-1 min-w-0"):
                                ui.label(d.title or "무제").classes("font-bold ellipsis")
                                ui.label((d.updated_at or d.created_at).strftime("%Y-%m-%d %H:%M")) \
                                    .classes("text-xs text-gray-500")
                            if inside:
                                ui.button("빼기", icon="remove", on_click=lambda d=d: vm.remove(d.id)) \
                                    .props("outline dense color=negative")
                            else:
                                ui.button("넣기", icon="add", on_click=lambda d=d: vm.add(d.id)) \
                                    .props("dense color=primary")
                            ui.button(icon="delete_forever", on_click=lambda d=d: _confirm_delete_lyric(vm, d)) \
                                .props("flat dense round color=negative").tooltip("가사 삭제")

            all_list()
            scope.refresh_on(all_list, "all_items", "members", "total")


def _confirm_delete_category(vm: LyricCategoriesViewModel) -> None:
    name = vm.category_name
    if not name:
        return
    with ui.dialog() as dialog, ui.card():
        ui.label(f"'{name}' 카테고리를 삭제할까요?").classes("font-bold")
        ui.label("카테고리 묶음만 없어지고, 안에 있던 가사는 그대로 남습니다.").classes("text-sm")
        with ui.row().classes("w-full justify-end"):
            ui.button("취소", on_click=dialog.close).props("flat")
            ui.button("삭제", on_click=lambda: (vm.delete_category(), dialog.close())).props("color=negative")
    dialog.open()


def _confirm_delete_lyric(vm: LyricCategoriesViewModel, d) -> None:  # type: ignore[no-untyped-def]
    with ui.dialog() as dialog, ui.card():
        ui.label(f"'{d.title or '무제'}' 가사를 삭제할까요?").classes("font-bold")
        ui.label("가사집과 모든 카테고리에서 지웁니다. 이 가사로 만든 곡은 남습니다.").classes("text-sm")
        with ui.row().classes("w-full justify-end"):
            ui.button("취소", on_click=dialog.close).props("flat")
            ui.button("삭제", on_click=lambda: (vm.delete_lyric(d.id), dialog.close())).props("color=negative")
    dialog.open()
