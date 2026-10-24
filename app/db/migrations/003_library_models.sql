-- 003: 음악 학습을 카테고리와 분리 (모델 라이브러리 + 카테고리 멤버십)
-- 1) datasets.category_id 를 NULL 허용으로 (카테고리 없는 '라이브러리' 데이터셋)
CREATE TABLE datasets_new (
    id            TEXT PRIMARY KEY,
    category_id   TEXT REFERENCES categories(id),
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
INSERT INTO datasets_new SELECT id, category_id, kind, version, status, source_path, json_path, tensor_path,
    rights_status, created_at FROM datasets;
DROP TABLE datasets;
ALTER TABLE datasets_new RENAME TO datasets;

-- 2) 카테고리 ↔ 모델 멤버십 (한 카테고리에 여러 모델, 한 모델을 여러 카테고리에)
CREATE TABLE IF NOT EXISTS category_model_members (
    category_id TEXT NOT NULL REFERENCES categories(id),
    model_id    TEXT NOT NULL REFERENCES models(id),
    added_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (category_id, model_id)
);

-- 기존 데이터 이전: 카테고리 소속 모델 + 이미 활성으로 연결된 모델을 멤버로
INSERT OR IGNORE INTO category_model_members (category_id, model_id)
    SELECT category_id, id FROM models WHERE category_id IS NOT NULL;
INSERT OR IGNORE INTO category_model_members (category_id, model_id)
    SELECT category_id, model_id FROM category_model_bindings
