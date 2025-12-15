# Data Model: AI-Native Book + RAG Chatbot

## Entities and Relationships

### BookContent Entity
```
Table: book_content
- id: UUID (Primary Key)
- title: VARCHAR(255) (Required)
- module_id: UUID (Foreign Key to modules table)
- section_path: VARCHAR(200) (Unique, e.g., "module-1-ros2/introduction")
- content_type: VARCHAR(50) (Enum: text, code, diagram, exercise, lab)
- content_body: TEXT (Required)
- metadata: JSONB (Optional, for tags, difficulty, prerequisites)
- version: INTEGER (Default: 1)
- created_at: TIMESTAMP (Auto-generated)
- updated_at: TIMESTAMP (Auto-generated)
```

### Module Entity
```
Table: modules
- id: UUID (Primary Key)
- name: VARCHAR(100) (Required, e.g., "Module 1: ROS 2 for Humanoid Robotics")
- description: TEXT (Optional)
- order_index: INTEGER (Required, for ordering modules)
- created_at: TIMESTAMP (Auto-generated)
- updated_at: TIMESTAMP (Auto-generated)
```

### Conversation Entity
```
Table: conversations
- id: UUID (Primary Key)
- user_id: VARCHAR(100) (Optional, for user identification)
- title: VARCHAR(200) (Optional, auto-generated if not provided)
- created_at: TIMESTAMP (Auto-generated)
- updated_at: TIMESTAMP (Auto-generated)
```

### UserQuery Entity
```
Table: user_queries
- id: UUID (Primary Key)
- conversation_id: UUID (Foreign Key to conversations table)
- query_text: TEXT (Required)
- selected_text: TEXT (Optional, text that was selected when query was made)
- timestamp: TIMESTAMP (Auto-generated)
- metadata: JSONB (Optional, for query context)
```

### ChatbotResponse Entity
```
Table: chatbot_responses
- id: UUID (Primary Key)
- user_query_id: UUID (Foreign Key to user_queries table)
- response_text: TEXT (Required)
- sources: JSONB (Array of source references with confidence scores)
- confidence_score: FLOAT (0.0-1.0, measure of response certainty)
- timestamp: TIMESTAMP (Auto-generated)
- metadata: JSONB (Optional, for AI model, tokens used, etc.)
```

### Embedding Entity (for vector store - Qdrant)
```
Collection: book_content_embeddings
- content_id: UUID (Reference to book_content.id)
- title: String (Content title)
- section_path: String (Content section path)
- content_body: String (Content body for context)
- embedding_vector: Array<Float> (1536-dimensional OpenAI embedding)
- metadata: Object (Additional metadata including source reference)
```

### APIKey Entity (for future authentication)
```
Table: api_keys
- id: UUID (Primary Key)
- user_id: VARCHAR(100) (Reference to user)
- key_hash: VARCHAR(255) (Hash of the API key)
- name: VARCHAR(100) (Descriptive name for the key)
- permissions: JSONB (Permissions associated with this key)
- created_at: TIMESTAMP (Auto-generated)
- expires_at: TIMESTAMP (Optional expiration)
- revoked_at: TIMESTAMP (When the key was revoked)
```

## Relationships
- Module (1) → (Many) BookContent
- Conversation (1) → (Many) UserQuery
- UserQuery (1) → (1) ChatbotResponse
- BookContent (1) → (Many) Embeddings (in Qdrant)

## Indexes
- BookContent: Index on (module_id, section_path) for efficient lookups
- BookContent: Index on content_type for filtering
- UserQuery: Index on conversation_id for conversation history retrieval
- ChatbotResponse: Index on user_query_id for query-response mapping
- Conversations: Index on user_id and updated_at for user history

## Constraints
- BookContent.section_path must be unique
- BookContent.module_id must reference an existing Module
- UserQuery.conversation_id must reference an existing Conversation
- ChatbotResponse.user_query_id must reference an existing UserQuery
- Content versioning: Updates create new versions rather than in-place changes
- Soft deletes for maintaining content history (using deleted_at timestamp)

## Data Flow
1. Book content is ingested and stored in book_content table
2. Content is processed into embeddings and stored in Qdrant
3. User queries are logged in user_queries table
4. AI responses with sources are stored in chatbot_responses table
5. Conversations group related queries and responses
6. All data is available for analytics and improvement

## Storage Considerations
- Large text content in content_body should be compressed if needed
- Metadata fields use JSONB for flexible schema
- Embeddings are stored in Qdrant for efficient similarity search
- Historical versions of content can be maintained for rollback capability
- Periodic archival of old conversations to optimize performance