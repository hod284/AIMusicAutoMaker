-- 007: 목소리 모델(Seed-VC) 엔진 추가 — models / category_model_bindings 의 engine CHECK 에 'voice' 허용
CREATE TABLE models_new (
    id                TEXT PRIMARY KEY,
    category_id       TEXT REFERENCES categories(id),
    engine            TEXT NOT NULL CHECK (engine IN ('qwen', 'ace', 'voice')),
    base_model        TEXT NOT NULL,
    adapter_path      TEXT NOT NULL,
    version           INTEGER NOT NULL,
    dataset_id        TEXT REFERENCES datasets(id),
    train_config_json TEXT NOT NULL DEFAULT '{}',
    file_hash         TEXT NOT NULL DEFAULT '',
    validation_status TEXT NOT NULL DEFAULT 'unverified',
    notes             TEXT NOT NULL DEFAULT '',
    created_at        TEXT NOT NULL,
    display_name      TEXT NOT NULL DEFAULT '',
    UNIQUE (category_id, engine, version)
);
INSERT INTO models_new (id, category_id, engine, base_model, adapter_path, version, dataset_id, train_config_json,
    file_hash, validation_status, notes, created_at, display_name)
    SELECT id, category_id, engine, base_model, adapter_path, version, dataset_id, train_config_json,
    file_hash, validation_status, notes, created_at, display_name FROM models;
DROP TABLE models;
ALTER TABLE models_new RENAME TO models;

CREATE TABLE category_model_bindings_new (
    category_id TEXT NOT NULL REFERENCES categories(id),
    engine      TEXT NOT NULL CHECK (engine IN ('qwen', 'ace', 'voice')),
    model_id    TEXT NOT NULL REFERENCES models(id),
    PRIMARY KEY (category_id, engine)
);
INSERT INTO category_model_bindings_new SELECT category_id, engine, model_id FROM category_model_bindings;
DROP TABLE category_model_bindings;
ALTER TABLE category_model_bindings_new RENAME TO category_model_bindings;
