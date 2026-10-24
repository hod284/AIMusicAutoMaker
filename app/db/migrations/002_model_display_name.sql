-- 002: 모델 레지스트리에 사용자가 지정하는 표시 이름 추가
ALTER TABLE models ADD COLUMN display_name TEXT NOT NULL DEFAULT ''
