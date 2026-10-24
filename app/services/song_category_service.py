"""만든 곡 분류 서비스: 곡 카테고리 만들기/삭제, 곡 넣기/빼기, 곡 삭제(DB + 파일)."""

from __future__ import annotations

from pathlib import Path

from app.core.errors import NotFoundError, ValidationError
from app.core.observer import EventBus, Topics
from app.domain.entities import Song
from app.repositories.content_repo import SongRepository
from app.repositories.song_category_repo import SongCategory, SongCategoryRepository

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".opus", ".ogg", ".aac", ".m4a"}


class SongCategoryService:
    def __init__(self, repo: SongCategoryRepository, songs: SongRepository, bus: EventBus | None = None) -> None:
        self.repo = repo
        self.song_repo = songs
        self.bus = bus or EventBus()

    def _changed(self, **info: object) -> None:
        self.bus.publish(Topics.SONG_CREATED, **info)   # 곡 목록 화면들이 이 토픽으로 새로고침

    # --- 카테고리 ---
    def list(self) -> list[SongCategory]:
        return self.repo.list()

    def create(self, name: str) -> SongCategory:
        name = name.strip()
        if not name:
            raise ValidationError("카테고리 이름을 입력하세요.")
        if len(name) > 40:
            raise ValidationError("카테고리 이름은 40자 이하로 입력하세요.")
        if self.repo.get_by_name(name):
            raise ValidationError(f"'{name}' 카테고리가 이미 있습니다.")
        cat = self.repo.add(name)
        self._changed(action="song_category_created")
        return cat

    def delete(self, category_id: str) -> None:
        self.repo.delete(category_id)
        self._changed(action="song_category_deleted")

    # --- 곡 넣기/빼기 ---
    def add_song(self, category_id: str, song_id: str) -> None:
        if not any(c.id == category_id for c in self.repo.list()):
            raise NotFoundError("카테고리를 찾을 수 없습니다.")
        self.repo.add_member(category_id, song_id)
        self._changed(action="song_member_added")

    def remove_song(self, category_id: str, song_id: str) -> None:
        self.repo.remove_member(category_id, song_id)
        self._changed(action="song_member_removed")

    # --- 조회 ---
    def list_songs(self, category_id: str | None, query: str = "") -> list[Song]:
        """category_id 가 없으면 전체 곡, 있으면 그 카테고리 곡만. 곡 제목으로 검색."""
        if category_id:
            return self.repo.songs_in(self.song_repo, category_id, query.strip())
        return self.song_repo.list(None, query.strip(), 1000, title_only=True)

    def membership(self) -> dict[str, list[str]]:
        return self.repo.categories_of()

    # --- 곡 삭제 ---
    def delete_song(self, song_id: str, *, delete_file: bool = True) -> str | None:
        """곡 기록 삭제. delete_file 이면 오디오 파일도 실제로 지운다. 반환: 지운 파일 경로."""
        song = self.song_repo.get(song_id)
        if song is None:
            raise NotFoundError("곡을 찾을 수 없습니다.", detail=song_id)
        removed: str | None = None
        if delete_file and song.audio_path:
            path = Path(song.audio_path)
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
                try:
                    path.unlink()
                except OSError as exc:
                    raise ValidationError(f"파일 삭제 실패: {exc}", detail=str(path)) from exc
                removed = str(path)
        self.repo.delete_song(song_id)
        self._changed(action="song_deleted", song_id=song_id)
        return removed
