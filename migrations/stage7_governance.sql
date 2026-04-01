-- Migration: Stage 7 — Governance
-- Creates audit_log table, retention_policy table, and adds soft-delete columns to users.
-- Run against: PostgreSQL at 43.157.203.211:31248, database=zeabur

-- ─────────────────────────────────────────────────────────
-- 1. audit_log table
-- ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS audit_log (
    id          VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     VARCHAR(36) NOT NULL,
    action      VARCHAR(100) NOT NULL,
    resource    VARCHAR(100) NOT NULL,
    resource_id VARCHAR(36),
    details     JSONB,
    ip_address  VARCHAR(50),
    user_agent  TEXT,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_user_id   ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action    ON audit_log(action);
CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_log(resource);
CREATE INDEX IF NOT EXISTS idx_audit_resource_id ON audit_log(resource_id);
CREATE INDEX IF NOT EXISTS idx_audit_created  ON audit_log(created_at DESC);


-- ─────────────────────────────────────────────────────────
-- 2. retention_policy table
-- ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS retention_policy (
    id             VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    resource       VARCHAR(50) NOT NULL UNIQUE,  -- 'conversations', 'messages', 'feedback'
    retention_days INT         NOT NULL DEFAULT 90,
    is_active      BOOLEAN     DEFAULT TRUE,
    updated_at     TIMESTAMP   DEFAULT NOW()
);

-- Seed default retention policies (idempotent — only inserts if resource is new)
INSERT INTO retention_policy (resource, retention_days, is_active)
VALUES
    ('conversations', 90, TRUE),
    ('messages',      90, TRUE),
    ('feedback',      90, TRUE)
ON CONFLICT (resource) DO NOTHING;


-- ─────────────────────────────────────────────────────────
-- 3. Soft-delete columns on users table
-- ─────────────────────────────────────────────────────────

ALTER TABLE users ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS deleted_at  TIMESTAMP;

-- Create index on is_deleted for efficient queries
CREATE INDEX IF NOT EXISTS idx_users_is_deleted ON users(is_deleted) WHERE is_deleted = TRUE;
