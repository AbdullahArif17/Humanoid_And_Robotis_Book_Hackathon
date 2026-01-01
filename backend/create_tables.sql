-- SQL Schema for AI-Native Book RAG Chatbot Application
-- Run this in your Neon database SQL editor to create all required tables

-- Create modules table first (referenced by other tables)
CREATE TABLE IF NOT EXISTS modules (
    id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    order_index INTEGER NOT NULL,
    estimated_duration_hours INTEGER,
    prerequisites VARCHAR,
    learning_objectives VARCHAR,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) UNIQUE NOT NULL,
    user_id VARCHAR(36),
    title VARCHAR(200),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conversation_metadata JSON
);

-- Create book_content table
CREATE TABLE IF NOT EXISTS book_content (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    module_id VARCHAR(50) REFERENCES modules(id),
    section_path VARCHAR(200) NOT NULL,
    content_type VARCHAR(20) NOT NULL,
    content_body TEXT NOT NULL,
    content_metadata JSON,
    version INTEGER DEFAULT 1,
    chunk_boundary BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create user_queries table
CREATE TABLE IF NOT EXISTS user_queries (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36),
    query_text TEXT NOT NULL,
    query_type VARCHAR(20) NOT NULL,
    conversation_id VARCHAR(36) REFERENCES conversations(id),
    selected_text TEXT,
    query_context JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create chatbot_responses table
CREATE TABLE IF NOT EXISTS chatbot_responses (
    id VARCHAR(36) PRIMARY KEY,
    query_id VARCHAR(36) NOT NULL REFERENCES user_queries(id),
    response_text TEXT NOT NULL,
    sources JSON,
    confidence_score FLOAT,
    tokens_used INTEGER,
    model_used VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id VARCHAR(36) PRIMARY KEY,
    content_id VARCHAR(36) REFERENCES book_content(id),
    chunk_text TEXT NOT NULL,
    chunk_metadata JSON,
    content_version INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create api_keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id VARCHAR(36) PRIMARY KEY,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100),
    user_id VARCHAR(36),
    is_active BOOLEAN DEFAULT TRUE,
    rate_limit_requests INTEGER DEFAULT 1000,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_user_queries_conversation_id ON user_queries(conversation_id);
CREATE INDEX IF NOT EXISTS idx_user_queries_timestamp ON user_queries(timestamp);
CREATE INDEX IF NOT EXISTS idx_book_content_module_id ON book_content(module_id);
CREATE INDEX IF NOT EXISTS idx_book_content_section_path ON book_content(section_path);