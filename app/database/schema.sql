-- ---------------------------------------------------------------------------
-- Lost & Found - reference schema (pgvector variant).
--
-- The application creates these tables through SQLAlchemy
-- (`scripts/seed_database.py`), so running this file is optional. It exists as
-- readable documentation of the shape the ORM produces.
--
-- On a server without pgvector the `embedding` columns become `REAL[]` and
-- similarity is computed in Python; everything else is identical.
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(120)  NOT NULL,
    email      VARCHAR(200)  NOT NULL UNIQUE,
    created_at TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS found_items (
    id              SERIAL PRIMARY KEY,
    category        VARCHAR(60)  NOT NULL,
    brand           VARCHAR(80),
    color           VARCHAR(60),
    description     TEXT         NOT NULL,
    -- Ownership evidence. Never selected by search, never shown to a user.
    hidden_features TEXT         NOT NULL DEFAULT '',
    location        VARCHAR(160) NOT NULL,
    found_time      TIMESTAMPTZ  NOT NULL,
    status          VARCHAR(30)  NOT NULL DEFAULT 'available',
    embedding       VECTOR(384),
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_found_items_category ON found_items (category);
CREATE INDEX IF NOT EXISTS ix_found_items_status   ON found_items (status);
CREATE INDEX IF NOT EXISTS ix_found_items_status_category ON found_items (status, category);

-- Approximate nearest-neighbour index for cosine distance.
CREATE INDEX IF NOT EXISTS ix_found_items_embedding
    ON found_items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);

CREATE TABLE IF NOT EXISTS lost_items (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER REFERENCES users (id) ON DELETE SET NULL,
    category    VARCHAR(60),
    brand       VARCHAR(80),
    color       VARCHAR(60),
    description TEXT         NOT NULL,
    location    VARCHAR(160),
    lost_time   TIMESTAMPTZ,
    raw_query   TEXT         NOT NULL DEFAULT '',
    status      VARCHAR(30)  NOT NULL DEFAULT 'searching',
    embedding   VECTOR(384),
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS claims (
    id                  SERIAL PRIMARY KEY,
    lost_item_id        INTEGER REFERENCES lost_items (id)  ON DELETE CASCADE,
    found_item_id       INTEGER NOT NULL REFERENCES found_items (id) ON DELETE CASCADE,
    confidence_score    DOUBLE PRECISION,
    confidence_level    VARCHAR(20),
    verification_status VARCHAR(20) NOT NULL DEFAULT 'pending',
    verification_score  DOUBLE PRECISION,
    user_answer         TEXT,
    reasons             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pickup_requests (
    id            SERIAL PRIMARY KEY,
    reference     VARCHAR(20)  NOT NULL UNIQUE,
    claim_id      INTEGER REFERENCES claims (id)        ON DELETE SET NULL,
    found_item_id INTEGER NOT NULL REFERENCES found_items (id) ON DELETE CASCADE,
    user_id       INTEGER REFERENCES users (id)         ON DELETE SET NULL,
    location      VARCHAR(160) NOT NULL,
    status        VARCHAR(30)  NOT NULL DEFAULT 'pending',
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS escalations (
    id                  SERIAL PRIMARY KEY,
    reference           VARCHAR(20) NOT NULL UNIQUE,
    user_request        TEXT        NOT NULL,
    candidate_ids       JSONB,
    confidence_scores   JSONB,
    verification_result VARCHAR(20),
    reason              TEXT        NOT NULL,
    status              VARCHAR(30) NOT NULL DEFAULT 'open',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS notifications (
    id         SERIAL PRIMARY KEY,
    channel    VARCHAR(30)  NOT NULL DEFAULT 'in_app',
    recipient  VARCHAR(200) NOT NULL,
    subject    VARCHAR(200) NOT NULL,
    body       TEXT         NOT NULL,
    created_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Help-desk knowledge base: one row per collection desk or FAQ entry.
-- Seeded rows are sample content (is_sample = TRUE); replace before real use.
CREATE TABLE IF NOT EXISTS help_articles (
    id            SERIAL PRIMARY KEY,
    slug          VARCHAR(80)  NOT NULL UNIQUE,
    kind          VARCHAR(20)  NOT NULL DEFAULT 'faq',   -- 'desk' | 'faq'
    title         VARCHAR(160) NOT NULL,
    body          TEXT         NOT NULL,
    desk_location VARCHAR(160),                         -- matches found_items.location
    hours         VARCHAR(160),
    phone         VARCHAR(60),
    email         VARCHAR(120),
    is_sample     BOOLEAN      NOT NULL DEFAULT TRUE,
    embedding     vector(384),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_help_articles_desk_location ON help_articles (desk_location);
