# AI Music Studio (studio/)

Mac 한 대에서 **Qwen3 + MLX-LM(가사)** 과 **ACE-Step 1.5(음악)** 를 하나의 NiceGUI 앱으로 묶는 로컬 작사·작곡 스튜디오입니다.
카테고리마다 가사 LoRA와 음악 LoRA를 독립적으로 연결하고, 작사 → 검수 → 음악 프롬프트 → 음악 생성 → 라이브러리까지 한 화면 흐름으로 처리합니다.

## 빠른 시작

```bash
cd ~/AI-Music-Studio/studio
cp .env.example .env                     # 필요 시 경로/포트 수정
.venv/bin/pip install -e ".[dev]"        # studio/.venv 에만 설치 (nicegui, httpx, pydantic, pytest)
bash scripts/verify_install.sh           # Phase 0: 설치본 읽기 전용 점검 → data/install_report/
.venv/bin/python main.py                 # http://127.0.0.1:8080
```

엔진 없이 UI만 볼 때: `STUDIO_MUSIC_ENGINE=mock STUDIO_LYRICS_ENGINE=mock STUDIO_VOICE_ENGINE=mock .venv/bin/python main.py`
(Mock 결과는 화면·DB에 `MOCK` 으로 표시되어 실제 생성물과 섞이지 않습니다.)

테스트: `.venv/bin/python -m pytest -q`

## 아키텍처: MVVM

```
View (NiceGUI, app/views)            화면만 그림. 로직 없음
  │  bind_value / ViewScope.on(prop)  ← ObservableProperty 변경 알림
  ▼
ViewModel (app/viewmodels)           화면 상태 + 커맨드. NiceGUI import 금지 → UI 없이 테스트
  │  EventBus 구독 (job.*, song.created …)
  ▼
Model
  ├─ services/      비즈니스 로직 (카테고리·가사·프롬프트 디렉터·음악·잡)
  ├─ adapters/      엔진 연결 (ACE REST, MLX 서브프로세스, Mock)
  ├─ repositories/  SQLite 접근
  └─ domain/        엔티티·열거형·상태머신
```

- View 는 VM 의 속성을 바인딩하고, 버튼은 VM 커맨드만 호출합니다.
- VM 은 서비스를 `ServiceContainer` 에서 받고, 서비스가 발행한 이벤트를 구독해 자신의 속성을 바꿉니다.
- 탭을 닫으면 `client.on_delete → vm.dispose()` 로 구독이 정리됩니다.

## 디자인 패턴 지도

| 패턴 | 위치 | 역할 |
|---|---|---|
| **Singleton** | `core/singleton.py` `SingletonMeta` | `AppSettings`, `Database`, `EventBus`, `ResourceLock`, `ServiceContainer` — 스레드 안전, 테스트용 `reset()` |
| **Adapter** | `adapters/ports.py` → `ace/rest_adapter.py`, `mlx/process_adapter.py`, `*/mock_adapter.py` | 서비스는 `MusicEnginePort` / `LyricsEnginePort` 에만 의존 |
| **Observer** | `core/observer.py` | `ObservableProperty`(VM→View 바인딩) + `EventBus`(서비스→VM 브로드캐스트, `job.*` 와일드카드) |
| Factory + Registry | `adapters/factory.py` | `STUDIO_MUSIC_ENGINE` (ace 또는 mock) 으로 구현 선택. 새 엔진은 `register_music()` 한 줄 |
| Command | `services/job_service.py` `JobHandler` | 작업 종류 하나 = 핸들러 클래스 하나 |
| Template Method | `JobManager._execute` | validating → queued → GPU 락 → running → 종료 골격 고정 |
| State | `domain/job_state.py` | 허용된 전이만 가능 (`created→…→succeeded/failed/cancelled`, 학습용 `exporting/validating_adapter`) |
| Strategy | `services/prompt_service.py` | `QwenDirectorStrategy` → 실패 시 `TemplateStrategy` |
| Chain of Responsibility | `Guardrail` 체인 | 빈 값 → 금지어 → 장르 이탈 → BPM 보정 → 보컬 조건 |
| Repository | `repositories/` | 서비스는 SQL 을 모름 |
| DI / Composition Root | `core/container.py` | 조립은 한 곳에서만. 테스트는 Mock 어댑터 주입 |

### 확장 예시

- **새 작업 종류**: `JobHandler` 서브클래스 작성 → `container.py` 의 `jobs.register(...)` 에 추가.
- **새 음악 엔진(예: ACE CLI fallback)**: `MusicEnginePort` 구현 → `AdapterFactory.register_music("ace-cli", builder)`.
- **새 가드레일 규칙**: `Guardrail` 서브클래스 → `default_guardrails()` 체인에 `.then(...)`.
- **새 화면**: VM(`BaseViewModel` 상속) + `@ui.page` View. VM 은 먼저 pytest 로 검증.

## ACE-Step 라우트: 추측 금지 원칙

`config/ace_routes.json` 이 "기능 → method/path" 바인딩입니다. 앱은 시작 시 설치본 `/openapi.json` 과 대조해 상태를 정합니다.

| 상태 | 의미 | UI |
|---|---|---|
| AVAILABLE | 바인딩 + 설치본 OpenAPI 모두 확인 | 사용 가능 |
| MISSING | 바인딩은 있으나 설치본에 없음 | 비활성 + 사유 |
| UNBOUND | 공식 문서에 경로/본문이 없어 비워 둠 | 비활성 + **후보 라우트만 제안** |

- 요청 본문은 OpenAPI 스키마의 필드만 남기고 나머지는 제거합니다. 스키마가 없으면 **공식 문서에 명시된 생성 필드**(`/release_task`, `/query_result`)만 fallback 으로 허용하고, 학습/LoRA 라우트는 본문을 만들지 않습니다.
- 공식 API 문서에는 LoRA load/unload 와 데이터셋/전처리 REST 경로가 없습니다. `verify_install.sh` 결과(설치본 OpenAPI + 라우터 grep)를 보고 바인딩을 채우세요. 필드명이 다르면 `field_map` (예: `{"adapter_path": "lora_path"}`)으로 매핑합니다.
- 설정 > 진단 화면에 기능 표와 설치본의 모든 라우트가 보입니다.

## 데이터 보호

- 폴더는 `workspace/categories/<category_id(UUID)>/...` — 이름을 바꿔도 경로 불변, 카테고리 간 혼합 불가.
- `PathGuard` 가 `../`·심볼릭 링크 탈출을 막고, `ACE-Step-1.5/datasets/test01`, `lora_output`, `checkpoints`, 두 `.venv` 를 **쓰기 금지**로 보호합니다.
- 기존 `final_lora` 는 카테고리·모델 화면에서 **참조 등록**만 합니다 (복사/수정 없음, 해시 기록).
- 파일 쓰기는 임시파일 → link/rename, 기본값은 덮어쓰기 거부. 학습 산출물은 항상 새 `vNNN` 폴더.
- 카테고리 삭제 대신 보관(복구 가능). 곡 파일은 DB 에 등록된 workspace 내부 경로만 `/media/songs/<id>` 로 제공.

## 구현 상태 (2026-10-24)

검증 환경: Linux 샌드박스, Python 3.13, NiceGUI 3.18 — **Mac/ACE/Qwen 실기 테스트는 아직 하지 않았습니다.**

| 영역 | 상태 | 검증 방식 |
|---|---|---|
| Phase 0 앱 기동·SQLite 마이그레이션·진단 화면 | 구현 | 실제 기동(7개 페이지 200) + pytest |
| Phase 1 카테고리 CRUD/보관/분리, 모델 레지스트리, 활성 바인딩 | 구현 | pytest (실제 SQLite·파일시스템) |
| Phase 2 Qwen 작사·편집·버전·검수, 음악 프롬프트 디렉터·가드레일 | 구현 | **Mock 엔진** pytest + UI 시뮬레이션 |
| Phase 3 ACE `/release_task`→`/query_result`→`/v1/audio`, LoRA 전환, A/B | 구현 | **httpx MockTransport**(문서 스키마 기반) + Mock 엔진 |
| Phase 4 가사 데이터셋 빌더(JSONL 검증·분할·버전), `mlx_lm.lora` 학습 잡 | 구현 | 옵션 매핑/로그 파싱 단위 테스트만. 실제 학습 미검증 |
| Phase 5 ACE 스캔·라벨·전처리·학습·Export | **인터페이스만** | 라우트 미확인 → 시작하지 않도록 차단 |
| Phase 6 원클릭 완성곡 (작사→프롬프트→음악) | 구현 | Mock 엔진 pytest (실패 시 가사 보존 확인) |
| Albums 화면 (P2) | DB 테이블만 | — |

### Mac 에서 먼저 확인할 것 (미해결)

1. `scripts/verify_install.sh` 실행 → `data/install_report/` 의 OpenAPI·라우터 grep 결과로 `config/ace_routes.json` 의 `lora_load / lora_unload / lora_status` 와 학습·데이터셋 라우트 채우기.
2. `/release_task` 에 `vocal_language: "ko"` 가 허용되는지 실제 응답으로 확인 (공식 문서 예시는 en/zh/ja 뿐).
3. `app/workers/mlx_generate_worker.py` 가 설치된 mlx_lm 버전의 `load/generate/make_sampler` 시그니처와 맞는지 (verify 스크립트 6단계).
4. `mlx_lm.lora` 산출물 파일명 (`adapters.safetensors` 가정) 확인 후 `category_service.QWEN_WEIGHT_CANDIDATES` 조정.
5. MLX DiT / MPS 경로에서 LoRA 가 실제 적용되는지는 로드 로그만으로 판단하지 말고 A/B 생성으로 확인.
6. ACE 서버 기동 스크립트가 `DYLD_LIBRARY_PATH`(FFmpeg 8)를 전달하는지 (verify 스크립트 4단계).

### 외부 RVC 보컬 모델

보컬 변환 → ② 바꿀 목소리 → **모델 넣기 (RVC)** 에서 `.pth`와 선택적인 `.index`를 올립니다.
업로드한 파일은 `workspace/library/voice/rvc`에 복사되어 다음에도 선택할 수 있습니다.
인덱스는 해당 모델과 함께 배포된 파일을 선택하세요. 인덱스가 없으면 자동으로 반영 비율 0으로 변환합니다.
RVC는 참고 녹음과 변환 스텝을 사용하지 않으며, 키 설정은 그대로 적용됩니다.

Applio는 Studio와 별도 Python 환경에 설치해야 합니다. 기본 설치 위치는 `voice-ai/Applio`입니다.
현재 설치본은 Applio `fb427aa388a954a387da7034fbbc08313f7ffe8c`의 `requirements-engine.txt`를 사용한 추론 전용 환경으로 구성되어 있습니다.
ContentVec와 RMVPE 공통 가중치도 설치되어 있습니다.
다른 위치에 설치했다면 `.env`의 `STUDIO_APPLIO_ROOT`와 `STUDIO_APPLIO_PYTHON`을 설정하고 Studio를 재시작하세요.
공식 저장소: https://github.com/IAHispano/Applio
Studio는 Applio `core.run_infer_script`를 별도 프로세스로 실행하며 실패·취소·시간 초과를 기존 작업 시스템에 전달합니다.
모델은 학습용 체크포인트가 아닌 RVC 추론용 `.pth`를 사용하세요. ZIP은 압축을 풀어서 올리세요.
