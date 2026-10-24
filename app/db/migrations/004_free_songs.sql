-- 004: 곡·가사를 카테고리 없이도 저장 (곡 만들기에서 모델을 직접 고르는 방식)
CREATE TABLE lyrics_drafts_new (
    id          TEXT PRIMARY KEY,
    category_id TEXT REFERENCES categories(id),
    lineage_id  TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    topic       TEXT NOT NULL DEFAULT '',
    lyrics      TEXT NOT NULL,
    version     INTEGER NOT NULL,
    source      TEXT NOT NULL,
    model_id    TEXT REFERENCES models(id),
    prompt_json TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    UNIQUE (lineage_id, version)
);
INSERT INTO lyrics_drafts_new SELECT id, category_id, lineage_id, title, topic, lyrics, version, source, model_id,
    prompt_json, created_at FROM lyrics_drafts;
DROP TABLE lyrics_drafts;
ALTER TABLE lyrics_drafts_new RENAME TO lyrics_drafts;
CREATE INDEX IF NOT EXISTS idx_lyrics_category ON lyrics_drafts(category_id);

CREATE TABLE songs_new (
    id              TEXT PRIMARY KEY,
    category_id     TEXT REFERENCES categories(id),
    lyrics_draft_id TEXT REFERENCES lyrics_drafts(id),
    music_model_id  TEXT REFERENCES models(id),
    qwen_model_id   TEXT REFERENCES models(id),
    title           TEXT NOT NULL DEFAULT '',
    audio_path      TEXT NOT NULL DEFAULT '',
    prompt          TEXT NOT NULL DEFAULT '',
    generation_json TEXT NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
INSERT INTO songs_new SELECT id, category_id, lyrics_draft_id, music_model_id, qwen_model_id, title, audio_path,
    prompt, generation_json, status, created_at FROM songs;
DROP TABLE songs;
ALTER TABLE songs_new RENAME TO songs;
CREATE INDEX IF NOT EXISTS idx_songs_category ON songs(category_id)
