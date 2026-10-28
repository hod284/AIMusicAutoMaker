"""음악 LoRA 학습 화면."""

from __future__ import annotations

import json
from typing import Any

from nicegui import ui

from app.domain.enums import JobKind
from app.viewmodels.music_training_vm import STATUS_KO, MusicModelsViewModel, MusicTrainingViewModel
from app.views.common import ViewScope, folder_picker, page_frame, progress_text, section, status_badge
from app.views.models_tab import models_tab

STEPS = ["① 업로드", "② 스캔", "③ 라벨링·검수", "④ 전처리", "⑤ 학습·Export"]
_FILE_KEYS = ("filename", "file_name", "audio_path", "file", "path", "name")


def _sample_title(sample: dict[str, Any], idx: int) -> str:
    for k in _FILE_KEYS:
        if sample.get(k):
            return str(sample[k]).rsplit("/", 1)[-1]
    return f"샘플 {idx + 1}"


@ui.page("/music-training")
async def music_training_page() -> None:
    vm = MusicTrainingViewModel()
    scope = ViewScope(vm)
    lockable: list[ui.element] = []      # 단계가 실행 중일 때 잠글 입력들

    def lock(el):  # type: ignore[no-untyped-def]
        lockable.append(el)
        return el

    def apply_lock(_v=None) -> None:  # type: ignore[no-untyped-def]
        for el in lockable:
            el.set_enabled(not vm.locked)
    with page_frame("음악 LoRA 학습"):
        with ui.tabs().classes("w-full") as tabs:
            t_train = ui.tab("학습", icon="model_training")
            t_models = ui.tab("모델 · 카테고리", icon="folder_special")
        with ui.tab_panels(tabs, value=t_train).classes("w-full bg-transparent"):
            with ui.tab_panel(t_train).classes("p-0 gap-4"):
                with ui.row().classes("w-full items-center gap-4"):
                    lock(ui.input("결과 모델 이름 *", placeholder="예: 가을 록발라드 v1").bind_value(vm, "model_title")
                         .props("maxlength=80 counter").classes("w-80"))
                    lock(ui.input("결과 저장 폴더", placeholder="비우면 기본 위치 (ACE-Step-1.5/datasets/studio/…)")
                         .bind_value(vm, "export_dir").classes("flex-1 min-w-[280px]"))
                    lock(ui.button("찾아보기", icon="folder_open", on_click=lambda: folder_picker(
                        vm.browse, vm.browse_root, vm, "export_dir", title="결과 저장 폴더 선택",
                        start=vm.export_dir or None)).props("outline"))

                    @ui.refreshable
                    def cap_warning() -> None:
                        if vm.missing_caps:
                            ui.label("ACE 서버에서 확인되지 않은 기능: " + ", ".join(vm.missing_caps)).classes(
                                "text-sm text-negative")

                    cap_warning()
                    scope.refresh_on(cap_warning, "capabilities")

                # --- 진행 단계 표시 ---
                @ui.refreshable
                def stepper() -> None:
                    with ui.row().classes("w-full gap-2"):
                        for i, name in enumerate(STEPS):
                            done = i < vm.step
                            ui.badge(name, color="positive" if done else ("primary" if i == vm.step else "grey")).classes(
                                "text-sm px-3 py-1")

                stepper()
                scope.refresh_on(stepper, "datasets", "dataset_id")

                # --- 작업 상태 ---
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
                                ui.button("중지", on_click=vm.cancel).props("flat dense color=negative")
                        if vm.losses:
                            ui.echart({
                                "xAxis": {"type": "value", "name": "step"}, "yAxis": {"type": "value", "name": "loss"},
                                "series": [{"type": "line", "data": [list(p) for p in vm.losses], "showSymbol": False}],
                                "tooltip": {"trigger": "axis"}, "grid": {"left": 50, "right": 20, "top": 30, "bottom": 30},
                            }).classes("w-full h-48")
                        ui.log(max_lines=300).classes("w-full h-32 text-xs").push("\n".join(vm.job_logs[-40:]) or "…")

                job_bar()
                scope.refresh_on(job_bar, "job", "job_logs", "losses")

                # --- ① 업로드 ---
                with section("① 오디오 업로드",
                             "같은 스타일의 곡만 모으세요. 학습·생성 권리가 있는 음원만 사용할 수 있습니다. (mp3/wav/flac/ogg/m4a)"):
                    async def on_upload(e) -> None:  # type: ignore[no-untyped-def]
                        vm.upload(e.file.name, await e.file.read())

                    lock(ui.upload(label="오디오 파일 선택 (여러 개 가능)", multiple=True, auto_upload=True,
                                   on_upload=on_upload, max_file_size=vm.svc.max_upload_bytes).props(
                        "accept=.mp3,.wav,.flac,.ogg,.m4a").classes("w-full"))

                    @ui.refreshable
                    def staged() -> None:
                        if not vm.staged:
                            ui.label("대기 중인 파일이 없습니다.").classes("text-xs text-gray-500")
                            return
                        ui.label(f"대기 중 {len(vm.staged)}개").classes("text-sm font-bold")
                        with ui.column().classes("w-full gap-0 max-h-48 overflow-y-auto"):
                            for name in vm.staged:
                                with ui.row().classes("items-center gap-2"):
                                    ui.icon("audio_file").classes("text-gray-500")
                                    ui.label(name).classes("text-sm")
                                    ui.button(icon="close", on_click=lambda n=name: vm.remove_staged(n)).props(
                                        "flat dense round size=sm").set_enabled(not vm.locked)

                    staged()
                    scope.refresh_on(staged, "staged", "job")

                    rights = {"ok": False, "note": ""}
                    lock(ui.input("출처 / 라이선스 메모", placeholder="예: 직접 녹음, FMA CC-BY 4.0 ...").bind_value(
                        rights, "note").classes("w-full"))
                    lock(ui.checkbox("이 음원들을 학습·생성에 사용할 권리가 있음을 확인했습니다.").bind_value(rights, "ok"))
                    lock(ui.button("데이터셋 버전 만들기", icon="create_new_folder",
                                   on_click=lambda: vm.create_dataset(rights["ok"], rights["note"])))

                # --- 데이터셋 선택 + 단계 실행 ---
                with section("②~⑤ 학습 단계", "각 단계는 ACE 서버에서 실행됩니다. 화면을 닫아도 계속 진행됩니다."):
                    @ui.refreshable
                    def dataset_select() -> None:
                        options = {d.id: f"{vm.dataset_label(d)} · {STATUS_KO.get(d.status, d.status)}"
                                   + ("" if d.rights_status == "confirmed" else " · 권리 미확인") for d in vm.datasets}
                        if not options:
                            ui.label("데이터셋이 없습니다. ① 에서 만들어 주세요.").classes("text-sm text-gray-500")
                            return
                        ui.select(options, label="데이터셋 버전", value=vm.dataset_id,
                                  on_change=lambda e: vm.choose_dataset(e.value)).classes("w-80") \
                            .set_enabled(not vm.locked)
                        if vm.dataset_id and not vm.server_has_dataset and vm.step >= 1:
                            ui.label("ACE 서버에 이 데이터셋이 올라가 있지 않습니다. 라벨링·검수·전처리 전에 ② 스캔을 다시 하세요."
                                     ).classes("text-xs text-orange-700")

                    dataset_select()
                    scope.refresh_on(dataset_select, "datasets", "dataset_id", "job")

                    actions = {JobKind.MUSIC_SCAN: ("② 스캔", "search", vm.scan),
                               JobKind.MUSIC_LABEL: ("③ 자동 라벨링", "label", vm.auto_label),
                               JobKind.MUSIC_PREPROCESS: ("④ 전처리", "memory", vm.preprocess),
                               JobKind.MUSIC_TRAIN: ("⑤ 학습 시작", "model_training", vm.train)}

                    def step_button(kind: JobKind) -> None:
                        """단계 버튼: 실행 중인 단계는 '… 중' + 로딩, 나머지는 잠김. 앞 단계를 안 했으면 잠김."""
                        name, icon, action = actions[kind]
                        if vm.running_step(kind):
                            ui.button(f"{name} 중…", icon="hourglass_top").props("disable").mark(f"step-{kind.value}")
                            ui.spinner("dots", size="md", color="primary")
                            return
                        btn = ui.button(name, icon=icon, on_click=action).mark(f"step-{kind.value}")
                        if kind != JobKind.MUSIC_TRAIN:
                            btn.props("outline")
                        btn.set_enabled(vm.step_ready(kind))
                        if not vm.locked and vm.dataset is not None and not vm.step_ready(kind) and vm.STEP_HINT[kind]:
                            btn.tooltip(vm.STEP_HINT[kind])

                    with ui.row().classes("items-end gap-3 w-full"):
                        lock(ui.input("트리거 태그 (선택)", placeholder="예: mystyle").bind_value(vm.scan_opts, "tag")
                             .classes("w-48"))
                        lock(ui.checkbox("전부 연주곡 (가사 없음)").bind_value(vm.scan_opts, "instrumental"))

                        @ui.refreshable
                        def step_buttons() -> None:
                            with ui.row().classes("items-center gap-3"):
                                for kind in (JobKind.MUSIC_SCAN, JobKind.MUSIC_LABEL, JobKind.MUSIC_PREPROCESS):
                                    step_button(kind)

                        step_buttons()
                        scope.refresh_on(step_buttons, "job", "datasets", "dataset_id")
                    with ui.row().classes("items-center gap-4"):
                        ui.label("라벨링 옵션:").classes("text-sm")
                        lock(ui.checkbox("가사 받아쓰기 (한국어 정확도 미검증)").bind_value(vm.label_opts, "transcribe_lyrics"))
                        lock(ui.checkbox("가사 형식 정리").bind_value(vm.label_opts, "format_lyrics"))
                        lock(ui.checkbox("라벨 없는 샘플만").bind_value(vm.label_opts, "only_unlabeled"))
                    ui.label("단계는 순서대로 열립니다 (② 스캔 → ③ 자동 라벨링 → ④ 전처리 → ⑤ 학습). "
                             "한 단계가 실행 중이면 끝날 때까지 다른 단계와 입력이 잠깁니다.").classes("text-xs text-gray-500")
                    ui.label("보컬곡과 연주곡이 섞여 있으면 '전부 연주곡'을 끄고, 검수에서 샘플별로 지정하세요. "
                             "처음 라벨링·전처리 때 ACE 모델을 자동으로 로드하므로 몇 분 걸릴 수 있습니다.").classes(
                        "text-xs text-gray-500")

                    # 단계 버튼 바로 아래: 지금 무엇이 실행 중인지 / 끝났는지
                    @ui.refreshable
                    def step_status() -> None:
                        job = vm.job
                        if job is None:
                            return
                        running = not job.status.is_terminal
                        color = {"succeeded": "bg-green-50 border-green-300", "failed": "bg-red-50 border-red-300",
                                 "cancelled": "bg-gray-100 border-gray-300"}.get(job.status.value, "bg-blue-50 border-blue-300")
                        with ui.card().classes(f"w-full border {color}"):
                            with ui.row().classes("items-center gap-3 w-full"):
                                if running:
                                    ui.spinner(size="md")
                                    ui.label(f"{vm.step_name} 진행 중…").classes("text-base font-bold text-primary")
                                elif job.status.value == "succeeded":
                                    ui.icon("check_circle", color="positive").classes("text-2xl")
                                    ui.label(f"{vm.step_name} 완료").classes("text-base font-bold text-positive")
                                elif job.status.value == "failed":
                                    ui.icon("error", color="negative").classes("text-2xl")
                                    ui.label(f"{vm.step_name} 실패").classes("text-base font-bold text-negative")
                                else:
                                    ui.icon("cancel", color="grey").classes("text-2xl")
                                    ui.label(f"{vm.step_name} 중지됨").classes("text-base font-bold")
                                ui.space()
                                ui.label(progress_text(job.progress) if running else "").classes("text-sm")
                                if running:
                                    ui.button("중지", on_click=vm.cancel).props("flat dense color=negative")
                            if running:
                                if job.progress is None:
                                    ui.linear_progress(show_value=False).props("indeterminate").classes("w-full")
                                else:
                                    ui.linear_progress(value=job.progress, show_value=False).classes("w-full")
                            detail = job.error if job.status.value == "failed" else vm.last_log
                            if detail:
                                ui.label(detail).classes("text-xs text-gray-600 break-all")

                    step_status()
                    scope.refresh_on(step_status, "job", "job_logs")

                    ui.separator()
                    ui.label("⑤ 학습 설정 (빈칸 = ACE 서버 기본값)").classes("text-sm font-bold")
                    cfg = vm.train_form
                    with ui.row().classes("items-end gap-3 w-full"):
                        lock(ui.number("Epoch", min=1, max=2000).bind_value(cfg, "train_epochs").classes("w-24"))
                        lock(ui.number("Learning rate", format="%.6f").bind_value(cfg, "learning_rate").classes("w-36"))
                        lock(ui.number("LoRA rank", min=1).bind_value(cfg, "lora_rank").classes("w-28"))
                        lock(ui.number("LoRA alpha", min=1).bind_value(cfg, "lora_alpha").classes("w-28"))
                        lock(ui.number("Batch", min=1).bind_value(cfg, "train_batch_size").classes("w-20"))
                        lock(ui.number("저장 주기(epoch)", min=1).bind_value(cfg, "save_every_n_epochs").classes("w-32"))
                    ui.label("결과는 '결과 저장 폴더/결과 모델 이름' 폴더에 저장되고(이미 있으면 덮어쓰지 않고 중단), "
                     "'모델·카테고리' 탭에 그 이름으로 나타납니다.").classes(
                        "text-xs text-gray-500")
                    @ui.refreshable
                    def train_button() -> None:
                        with ui.row().classes("items-center gap-3"):
                            step_button(JobKind.MUSIC_TRAIN)

                    train_button()
                    scope.refresh_on(train_button, "job", "datasets", "dataset_id")
                    ui.label("첫 실험은 10~20 epoch 로 끝까지 도는지 먼저 확인하세요. loss 가 내려가도 음악 품질은 따로 들어봐야 합니다."
                             ).classes("text-xs text-gray-500")

                # --- ③ 검수 ---
                with section("③ 샘플 검수", "자동 라벨은 AI 추정값이라 틀릴 수 있습니다. 가사·연주곡 여부를 확인하고 고친 뒤 전처리하세요."):
                    lock(ui.button("샘플 다시 불러오기", icon="refresh", on_click=vm.reload_samples).props("flat dense"))

                    @ui.refreshable
                    def samples() -> None:
                        if not vm.samples:
                            ui.label("스캔 후 샘플이 여기에 표시됩니다.").classes("text-xs text-gray-500")
                            return
                        with ui.column().classes("w-full gap-2 max-h-[70vh] overflow-y-auto"):
                            for idx, s in enumerate(vm.samples):
                                _sample_card(vm, idx, s)

                    samples()
                    scope.refresh_on(samples, "samples")
                    lock_state = {"locked": vm.locked}

                    def samples_on_lock(_v=None) -> None:  # type: ignore[no-untyped-def]
                        if lock_state["locked"] != vm.locked:      # 진행률 갱신마다가 아니라 잠김이 바뀔 때만 다시 그림
                            lock_state["locked"] = vm.locked
                            samples.refresh()

                    scope.on("job", samples_on_lock)

            with ui.tab_panel(t_models).classes("p-0"):
                models_tab(MusicModelsViewModel())
    apply_lock()
    scope.on("job", apply_lock)
    ui.timer(0.1, vm.load_capabilities, once=True)   # 페이지를 먼저 보여 주고 ACE 기능 확인 (생성 중엔 응답이 늦음)


# PUT /v1/dataset/sample/{idx} 가 받는 필드 (설치본 스키마 확인)
_EDITABLE = ("caption", "genre", "lyrics", "bpm", "keyscale", "timesignature", "language", "is_instrumental")


def _sample_card(vm: MusicTrainingViewModel, idx: int, s: dict[str, Any]) -> None:
    edit = {k: s.get(k) for k in _EDITABLE if k in s}
    card = ui.card().classes("w-full")
    with card:
        with ui.row().classes("items-center gap-2 w-full"):
            ui.label(f"{idx + 1}. {_sample_title(s, idx)}").classes("font-bold")
            if s.get("duration") not in (None, ""):
                ui.badge(f"duration: {s['duration']}", color="grey")
        if "caption" in edit:
            ui.input("캡션").bind_value(edit, "caption").classes("w-full")
        with ui.row().classes("gap-2 w-full"):
            if "genre" in edit:
                ui.input("장르").bind_value(edit, "genre").classes("w-40")
            if "bpm" in edit:
                ui.number("BPM", min=30, max=300).bind_value(edit, "bpm").classes("w-24")
            if "keyscale" in edit:
                ui.input("키").bind_value(edit, "keyscale").classes("w-28")
            if "timesignature" in edit:
                ui.input("박자").bind_value(edit, "timesignature").classes("w-20")
            if "language" in edit:
                ui.input("언어").bind_value(edit, "language").classes("w-24")
        if "lyrics" in edit:
            ui.textarea("가사 ([Instrumental] = 연주곡)").bind_value(edit, "lyrics").props("autogrow").classes(
                "w-full font-mono text-xs")
        if "is_instrumental" in edit:
            ui.checkbox("연주곡").bind_value(edit, "is_instrumental")
        if edit:
            ui.button("이 샘플 저장", icon="save",
                      on_click=lambda i=idx, e=edit: vm.save_sample(
                          i, {k: (int(v) if k == "bpm" and v not in (None, "") else v) for k, v in e.items()
                              if not (k == "bpm" and v in (None, ""))})).props("flat dense")
        else:
            ui.label("수정 가능한 필드가 응답에 없습니다.").classes("text-xs text-gray-500")
        with ui.expansion("원본 응답").classes("w-full text-xs"):
            ui.code(json.dumps(s, ensure_ascii=False, indent=2), language="json").classes("w-full")
    if vm.locked:                       # 단계 실행 중에는 검수 수정도 잠금
        for el in card.descendants():
            if isinstance(el, (ui.input, ui.number, ui.textarea, ui.checkbox, ui.button)):
                el.set_enabled(False)
