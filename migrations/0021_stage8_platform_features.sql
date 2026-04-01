-- Migration: Stage 8 Platform Features
-- Adds share_tokens, connectors_config tables and full-text search indexes

-- ─────────────────────────────────────────────────────────────────────────────
-- share_tokens — read-only share links for conversations
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS share_tokens (
    id              VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id VARCHAR(36) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    token           VARCHAR(64) UNIQUE NOT NULL,
    created_by      VARCHAR(36) NOT NULL REFERENCES users(id),
    expires_at      TIMESTAMP,
    is_active       BOOLEAN DEFAULT TRUE,
    view_count      INT DEFAULT 0,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_share_token ON share_tokens(token);
CREATE INDEX IF NOT EXISTS idx_share_conversation ON share_tokens(conversation_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- connectors_config — persistent configuration for external data connectors
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS connectors_config (
    id             VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid(),
    connector_type VARCHAR(50) NOT NULL UNIQUE,  -- 'filesystem', 's3', 'googledrive', 'notion'
    config          JSONB NOT NULL,               -- connection credentials, paths, etc.
    is_active       BOOLEAN DEFAULT FALSE,
    created_by      VARCHAR(36) REFERENCES users(id),
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Full-text search indexes
-- Use ` USING gin(...)` for fast tsvector lookups.
-- These indexes enable PostgreSQL full-text search on titles and message content.
-- ─────────────────────────────────────────────────────────────────────────────

-- Index on conversations.title for title search
CREATE INDEX IF NOT EXISTS idx_conversations_title_fts
    ON conversations
    USING gin(to_tsvector('english', coalesce(title, '')));

-- Index on messages.content for message body search
CREATE INDEX IF NOT EXISTS idx_messages_content_fts
    ON messages
    USING gin(to_tsvector('english', coalesce(content, '')));
