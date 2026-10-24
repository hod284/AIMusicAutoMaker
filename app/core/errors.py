"""도메인 예외. 모든 예외는 사용자용 한국어 메시지와 진단용 상세 정보를 분리해 갖는다."""

from __future__ import annotations

from typing import Any


class StudioError(Exception):
    """앱 공통 예외의 베이스."""

    user_message: str = "작업 중 오류가 발생했습니다."

    def __init__(self, user_message: str | None = None, *, detail: Any = None) -> None:
        self.user_message = user_message or self.user_message
        self.detail = detail
        super().__init__(self.user_message if detail is None else f"{self.user_message} | {detail}")


class ValidationError(StudioError):
    user_message = "입력값이 올바르지 않습니다."


class NotFoundError(StudioError):
    user_message = "대상을 찾을 수 없습니다."


class PathSecurityError(StudioError):
    user_message = "허용되지 않은 경로입니다."


class EngineUnavailableError(StudioError):
    user_message = "AI 엔진에 연결할 수 없습니다."


class CapabilityUnsupportedError(StudioError):
    user_message = "현재 설치된 엔진 버전에서 지원하지 않는 기능입니다."


class EngineResponseError(StudioError):
    user_message = "AI 엔진이 예상하지 못한 응답을 반환했습니다."


class ResourceBusyError(StudioError):
    user_message = "다른 GPU 작업이 실행 중입니다. 잠시 후 다시 시도하세요."


class InvalidStateTransition(StudioError):
    user_message = "작업 상태를 변경할 수 없습니다."
