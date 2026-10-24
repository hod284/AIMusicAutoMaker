-- 만든 곡 분류용 카테고리 (모델 카테고리와 별개). 곡 하나가 여러 카테고리에 들어갈 수 있다.
CREATE TABLE IF NOT EXISTS song_categories (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS song_category_members (
    category_id TEXT NOT NULL REFERENCES song_categories(id),
    song_id     TEXT NOT NULL REFERENCES songs(id),
    added_at    TEXT NOT NULL,
    PRIMARY KEY (category_id, song_id)
);
CREATE INDEX IF NOT EXISTS idx_song_members_song ON song_category_members(song_id);
