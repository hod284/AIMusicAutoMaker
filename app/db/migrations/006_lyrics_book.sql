-- 006: 가사집 — 가사 하나 = 기록 하나 (저장하면 덮어씀), 가사 카테고리 (곡·모델 카테고리와 별개)
-- 예전 v1/v2/v3 기록은 지우지 않는다. 목록에는 묶음마다 최신 버전 하나만 보이고 그 기록에 덮어쓴다.
ALTER TABLE lyrics_drafts ADD COLUMN updated_at TEXT;
UPDATE lyrics_drafts SET updated_at = created_at WHERE updated_at IS NULL;

CREATE TABLE IF NOT EXISTS lyric_categories (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lyric_category_members (
    category_id TEXT NOT NULL REFERENCES lyric_categories(id),
    lyric_id    TEXT NOT NULL REFERENCES lyrics_drafts(id),
    added_at    TEXT NOT NULL,
    PRIMARY KEY (category_id, lyric_id)
);
CREATE INDEX IF NOT EXISTS idx_lyric_members_lyric ON lyric_category_members(lyric_id);
