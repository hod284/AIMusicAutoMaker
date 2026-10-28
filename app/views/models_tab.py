"""'모델 · 카테고리' 탭 (음악 LoRA / 가사 LoRA 공통 화면)."""

from __future__ import annotations

from nicegui import ui

from app.viewmodels.models_vm import ModelsViewModel
from app.views.common import ViewScope


def models_tab(vm: ModelsViewModel) -> None:
    """만든 LoRA 를 카테고리에 넣고 빼는 화면. 왼쪽: 카테고리 / 오른쪽: 전체 모델."""
    kind = vm.kind_label
    scope = ViewScope(vm)
    with ui.row().classes("w-full gap-4 items-start no-wrap"):
        # --- 왼쪽: 카테고리와 그 안의 모델 ---
        with ui.card().classes("w-[440px] shrink-0"):
            ui.label("카테고리").classes("font-bold")
            new_cat = {"name": ""}
            with ui.row().classes("w-full items-end gap-2"):
                ui.input("새 카테고리 이름").bind_value(new_cat, "name").classes("flex-1")
                ui.button("카테고리 추가", icon="add",
                          on_click=lambda: (vm.create_category(new_cat["name"]), new_cat.update(name=""))
                          ).props("outline dense")

            @ui.refreshable
            def category_panel() -> None:
                if not vm.categories:
                    ui.label("카테고리가 없습니다. 위에서 만드세요.").classes("text-xs text-gray-500")
                    return
                with ui.row().classes("w-full items-end gap-2 no-wrap"):
                    ui.select({c.id: c.name for c in vm.categories}, label="편집할 카테고리", value=vm.category_id,
                              on_change=lambda e: vm.select_category(e.value)).classes("flex-1")
                    ui.button(icon="delete", on_click=lambda: _confirm_delete_category(vm)) \
                        .props("flat round color=negative").tooltip("카테고리 삭제 (모델은 남음)")
                ui.label(f"이 카테고리의 {kind} {len(vm.members)}개").classes("text-sm font-bold mt-2")
                with ui.column().classes("w-full gap-1 max-h-[55vh] overflow-y-auto"):
                    if not vm.members:
                        ui.label("오른쪽 목록에서 [넣기]로 모델을 추가하세요.").classes("text-xs text-gray-500")
                    for m in vm.members:
                        in_use = m.id == vm.active_id
                        with ui.card().classes("w-full p-2 " + ("border border-green-400 bg-green-50" if in_use else "")):
                            with ui.row().classes("items-center gap-2 w-full no-wrap"):
                                ui.label(m.label).classes("font-bold flex-1 ellipsis")
                                if in_use:
                                    ui.badge("사용 중", color="positive")
                                else:
                                    ui.button("사용", on_click=lambda m=m: vm.use(m.id)).props("flat dense")
                                ui.button(icon="remove_circle_outline", on_click=lambda m=m: vm.remove(m.id)) \
                                    .props("flat dense round color=negative").tooltip("카테고리에서 빼기")
                if vm.active_id:
                    ui.button(vm.none_label, on_click=lambda: vm.use(None)).props("flat dense size=sm")
                ui.label(vm.use_hint).classes(
                    "text-xs text-gray-500")

            category_panel()
            scope.refresh_on(category_panel, "categories", "category_id", "members", "active_id")

        # --- 오른쪽: 전체 모델 ---
        with ui.card().classes("flex-1 min-w-0"):
            ui.label(f"만든 {kind} 전체").classes("font-bold")
            search = ui.input(placeholder="모델 이름·경로 검색", on_change=lambda e: vm.set_search(e.value)) \
                .props("dense outlined clearable").classes("w-full")
            with search.add_slot("prepend"):
                ui.icon("search")

            @ui.refreshable
            def model_list() -> None:
                items = vm.visible_models
                ui.label(f"{len(items)} / {len(vm.models)}개").classes("text-xs text-gray-500")
                with ui.column().classes("w-full gap-2 max-h-[65vh] overflow-y-auto"):
                    if not items:
                        ui.label("아직 만든 모델이 없습니다. '학습' 탭에서 학습하세요.").classes("text-xs text-gray-500")
                    for m in items:
                        member = m.id in vm.member_ids
                        with ui.card().classes("w-full p-3"):
                            with ui.row().classes("items-center gap-2 w-full no-wrap"):
                                ui.label(m.label).classes("font-bold flex-1 ellipsis")
                                if member:
                                    ui.button("빼기", icon="remove", on_click=lambda m=m: vm.remove(m.id)) \
                                        .props("outline dense color=negative")
                                else:
                                    ui.button("넣기", icon="add", on_click=lambda m=m: vm.add(m.id)) \
                                        .props("dense color=primary")
                                ui.button(icon="delete_forever", on_click=lambda m=m: _confirm_delete_model(vm, m)) \
                                    .props("flat dense round color=negative").tooltip("모델 삭제 (DB + 파일)")
                            cats = vm.category_names_of(m.id)
                            with ui.row().classes("gap-1 items-center"):
                                ui.label(f"v{m.version} · {m.created_at.strftime('%Y-%m-%d %H:%M')}").classes(
                                    "text-xs text-gray-500")
                                for name in cats:
                                    ui.badge(name, color="grey")
                            ui.label(m.adapter_path).classes("text-xs text-gray-400 break-all")

            model_list()
            scope.refresh_on(model_list, "models", "model_search", "members", "categories")


def _confirm_delete_model(vm: ModelsViewModel, model) -> None:  # type: ignore[no-untyped-def]
    targets = vm.delete_targets(model.id)
    if targets is None:          # 경로 검사 실패 → 오류 알림은 VM 이 띄움
        return
    opt = {"files": True}
    with ui.dialog() as dialog, ui.card().classes("w-[640px] max-w-full"):
        ui.label(f"'{model.label}' 모델을 삭제할까요?").classes("font-bold")
        ui.label("앱 목록(DB)에서 지웁니다. 이 모델로 만든 곡은 남습니다.").classes("text-sm")
        if targets:
            ui.checkbox("아래 폴더도 실제로 삭제 (휴지통 아님, 되돌릴 수 없음)").bind_value(opt, "files")
            with ui.column().classes("w-full gap-0 bg-red-50 rounded p-2"):
                for t in targets:
                    ui.label(t).classes("text-xs font-mono break-all")
        else:
            ui.label("이 앱이 만든 파일 폴더가 없어 파일은 건드리지 않습니다.").classes("text-xs text-gray-500")
        with ui.row().classes("w-full justify-end"):
            ui.button("취소", on_click=dialog.close).props("flat")
            ui.button("삭제", icon="delete_forever",
                      on_click=lambda: (vm.delete_model(model.id, opt["files"]), dialog.close())) \
                .props("color=negative")
    dialog.open()


def _confirm_delete_category(vm: ModelsViewModel) -> None:
    name = next((c.name for c in vm.categories if c.id == vm.category_id), "")
    if not name:
        return
    with ui.dialog() as dialog, ui.card():
        ui.label(f"'{name}' 카테고리를 삭제할까요?").classes("font-bold")
        ui.label("카테고리 묶음만 없어지고, 안에 있던 모델은 그대로 남습니다.").classes("text-sm")
        with ui.row():
            ui.button("삭제", on_click=lambda: (vm.delete_category(), dialog.close())).props("color=negative")
            ui.button("취소", on_click=dialog.close).props("flat")
    dialog.open()
