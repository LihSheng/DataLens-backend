-- Migration: Stage 6 — Feedback + Evaluation tables
-- Target: PostgreSQL at 43.157.203.211:31248, database=zeabur
-- Run with: psql "postgresql://zeabur:bWCn0KkD39FlEuea8p5H1f7I4yMq6zO2@43.157.203.211:31248/zeabur" -f migrate_stage6.sql

BEGIN;

-- ─────────────────────────────────────────────────────────
-- feedback table
-- Stores user feedback (votes, ratings, comments) on RAG answers
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    message_id      UUID NOT NULL,
    user_id         UUID NOT NULL,
    vote            VARCHAR(10),          -- 'positive' or 'negative'
    rating          INT CHECK (rating BETWEEN 1 AND 5),
    comment         TEXT,
    metadata_json   TEXT,                 -- JSON blob for extensibility
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_message_id ON feedback(message_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id   ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at DESC);

-- ─────────────────────────────────────────────────────────
-- golden_dataset table
-- Ground-truth Q&A pairs for RAG evaluation
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS golden_dataset (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    context     TEXT[],                  -- list of source context strings
    source      VARCHAR(100),            -- e.g. 'manual', 'extracted'
    tags        TEXT[],
    created_by  UUID,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_golden_question ON golden_dataset USING gin(to_tsvector('english', question));
CREATE INDEX IF NOT EXISTS idx_golden_source   ON golden_dataset(source);
CREATE INDEX IF NOT EXISTS idx_golden_tags     ON golden_dataset USING gin(tags);
CREATE INDEX IF NOT EXISTS idx_golden_created  ON golden_dataset(created_at DESC);

-- ─────────────────────────────────────────────────────────
-- experiments table
-- Experiment runs with configurable retrieval/chain settings
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiments (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            VARCHAR(255) NOT NULL,
    description     TEXT,
    config_json     TEXT NOT NULL,       -- JSON: retrieval config, model settings
    results_json    TEXT,                -- JSON: aggregated RAGAS scores
    status          VARCHAR(50) DEFAULT 'pending',
                                          -- pending | running | completed | failed
    created_by      UUID,
    created_at      TIMESTAMP DEFAULT NOW(),
    completed_at    TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_experiments_status    ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created_by ON experiments(created_by);
CREATE INDEX IF NOT EXISTS idx_experiments_created  ON experiments(created_at DESC);

-- ─────────────────────────────────────────────────────────
-- experiment_results table
-- Individual question-level evaluation results
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiment_results (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id       UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    question            TEXT NOT NULL,
    expected_answer     TEXT,
    generated_answer    TEXT,
    retrieved_contexts  TEXT[],          -- list of context strings
    metrics_json        TEXT,             -- JSON: RAGAS scores
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_experiment_results_experiment_id
    ON experiment_results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_experiment_results_created
    ON experiment_results(created_at DESC);

COMMIT;

-- ─────────────────────────────────────────────────────────
-- Seed: Example golden Q&A entries (optional — remove for production)
-- ─────────────────────────────────────────────────────────
-- INSERT INTO golden_dataset (question, answer, context, source, tags, created_by)
-- VALUES
-- (
--     'What is RAG and how does it work?',
--     'Retrieval-Augmented Generation (RAG) combines a retrieval system with a generative LLM. '
--     'It retrieves relevant documents from a knowledge base and uses them as context to generate answers.',
--     ARRAY[
--         'RAG was introduced by Facebook AI Research in 2020 as a method to augment language models '
--         'with external knowledge.',
--         'A RAG system typically consists of an encoder for documents, a retriever, and a generative model.'
--     ],
--     'manual',
--     ARRAY['foundation', 'rag'],
--     NULL
-- ),
-- (
--     'What are the main components of a RAG pipeline?',
--     'A RAG pipeline consists of: 1) Document ingestion and chunking, '
--     '2) Embedding generation and vector storage, 3) Retrieval of relevant chunks, '
--     '4) Context assembly and 5) Generation by an LLM.',
--     ARRAY[
--         'The ingestion pipeline breaks documents into smaller chunks, generates embeddings, '
--         'and stores them in a vector database.',
--         'During query time, the retriever finds the k most relevant chunks which are then '
--         'passed to the LLM as context.'
--     ],
--     'manual',
--     ARRAY['pipeline', 'architecture'],
--     NULL
-- );
