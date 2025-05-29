CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


CREATE TABLE plan_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_type text NOT NULL,
    content text NOT NULL,
    section_path jsonb,
    embedding vector(384)
);