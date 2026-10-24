-- 001: 초기 스키마 (REQUIREMENTS 10장)
CREATE TABLE IF NOT EXISTS categories (
    id                     TEXT PRIMARY KEY,
    slug                   TEXT NOT NULL UNIQUE,
    name                   TEXT NOT NULL,
    description            TEXT NOT NULL DEFAULT '',
    language               TEXT NOT NULL DEFAULT 'ko',
    tags_json              TEXT NOT NULL DEFAULT '[]',
    status                 TEXT NOT NULL DEFAULT 'active',
    style_constraints_json TEXT NOT NULL DEFAULT '{}',
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS licenses (
    id            TEXT PRIMARY KEY,
    source_url    TEXT NOT NULL DEFAULT '',
    creator       TEXT NOT NULL DEFAULT '',
    license_name  TEXT NOT NULL DEFAULT '',
    scope         TEXT NOT NULL DEFAULT '',
    evidence_path TEXT NOT NULL DEFAULT '',
    verified_at   TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    id            TEXT PRIMARY KEY,
    category_id   TEXT NOT NULL REFERENCES categories(id),
    kind          TEXT NOT NULL CHECK (kind IN ('lyrics', 'music')),
    version       INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'draft',
    source_path   TEXT NOT NULL DEFAULT '',
    json_path     TEXT NOT NULL DEFAULT '',
    tensor_path   TEXT NOT NULL DEFAULT '',
    rights_status TEXT NOT NULL DEFAULT 'unverified',
    created_at    TEXT NOT NULL,
    UNIQUE (category_id, kind, version)
);

CREATE TABLE IF NOT EXISTS dataset_items (
    id              TEXT PRIMARY KEY,
    dataset_id      TEXT NOT NULL REFERENCES datasets(id),
    source_path     TEXT NOT NULL DEFAULT '',
    sha256          TEXT NOT NULL DEFAULT '',
    caption         TEXT NOT NULL DEFAULT '',
    lyrics          TEXT NOT NULL DEFAULT '',
    is_instrumental INTEGER NOT NULL DEFAULT 0,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    license_id      TEXT REFERENCES licenses(id),
    lyrics_source   TEXT,
    UNIQUE (dataset_id, sha256)
);

CREATE TABLE IF NOT EXISTS models (
    id                TEXT PRIMARY KEY,
    category_id       TEXT REFERENCES categories(id),
    engine            TEXT NOT NULL CHECK (engine IN ('qwen', 'ace')),
    base_model        TEXT NOT NULL,
    adapter_path      TEXT NOT NULL,
    version           INTEGER NOT NULL,
    dataset_id        TEXT REFERENCES datasets(id),
    train_config_json TEXT NOT NULL DEFAULT '{}',
    file_hash         TEXT NOT NULL DEFAULT '',
    validation_status TEXT NOT NULL DEFAULT 'unverified',
    notes             TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    UNIQUE (category_id, engine, version)
);

-- 카테고리×엔진 당 활성 모델은 정확히 0~1개 (PK 로 강제)
CREATE TABLE IF NOT EXISTS category_model_bindings (
    category_id TEXT NOT NULL REFERENCES categories(id),
    engine      TEXT NOT NULL CHECK (engine IN ('qwen', 'ace')),
    model_id    TEXT NOT NULL REFERENCES models(id),
    PRIMARY KEY (category_id, engine)
);

CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    category_id  TEXT REFERENCES categories(id),
    dataset_id   TEXT REFERENCES datasets(id),
    model_id     TEXT REFERENCES models(id),
    status       TEXT NOT NULL,
    progress     REAL,
    params_json  TEXT NOT NULL DEFAULT '{}',
    result_json  TEXT NOT NULL DEFAULT '{}',
    error        TEXT,
    retries      INTEGER NOT NULL DEFAULT 0,
    heartbeat_at TEXT,
    created_at   TEXT NOT NULL,
    started_at   TEXT,
    ended_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_category ON jobs(category_id);

CREATE TABLE IF NOT EXISTS job_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       TEXT NOT NULL REFERENCES jobs(id),
    timestamp    TEXT NOT NULL,
    level        TEXT NOT NULL,
    message      TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id);

CREATE TABLE IF NOT EXISTS lyrics_drafts (
    id          TEXT PRIMARY KEY,
    category_id TEXT NOT NULL REFERENCES categories(id),
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
CREATE INDEX IF NOT EXISTS idx_lyrics_category ON lyrics_drafts(category_id);

CREATE TABLE IF NOT EXISTS songs (
    id              TEXT PRIMARY KEY,
    category_id     TEXT NOT NULL REFERENCES categories(id),
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
CREATE INDEX IF NOT EXISTS idx_songs_category ON songs(category_id);

CREATE TABLE IF NOT EXISTS albums (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    concept    TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS album_tracks (
    album_id    TEXT NOT NULL REFERENCES albums(id),
    song_id     TEXT NOT NULL REFERENCES songs(id),
    track_order INTEGER NOT NULL,
    PRIMARY KEY (album_id, song_id),
    UNIQUE (album_id, track_order)
);
