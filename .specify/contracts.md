# Contracts: AI-Native Book + RAG Chatbot

## API Contracts

### Backend API Contract
- Base URL: `/api/v1/`
- Content-Type: `application/json`
- Response format: `{ "data": ..., "status": "success", "timestamp": "..." }`
- Error format: `{ "error": "message", "code": "error_code", "details": {...} }`

### Content Management Endpoints
- `POST /content/` - Create new content
  - Request: `{ "title": "string", "module_id": "string", "content_type": "enum", "body": "string" }`
  - Response: `{ "id": "uuid", "created_at": "timestamp", ... }`
- `GET /content/{id}` - Retrieve content
- `PUT /content/{id}` - Update content
- `DELETE /content/{id}` - Delete content

### Chat Endpoints
- `POST /chat/conversations/` - Create conversation
- `POST /chat/conversations/{id}/query` - Process query with RAG
  - Request: `{ "query": "string", "selected_text": "string?" }`
  - Response: `{ "response": "string", "sources": [...], "confidence": "float" }`
- `GET /chat/conversations/{id}/history` - Get conversation history

## Data Contracts

### BookContent Entity
- `id`: UUID (required, immutable)
- `title`: String (1-255 chars, required)
- `module_id`: String (foreign key to Module, required)
- `section_path`: String (unique path like "module-1/ros2/intro", required)
- `content_type`: Enum (text, code, diagram, exercise, lab, required)
- `content_body`: Text (required, min 1 char)
- `metadata`: JSON (optional, for tags, difficulty, etc.)
- `version`: Integer (required, starts at 1)
- `created_at`: Timestamp (auto-generated)
- `updated_at`: Timestamp (auto-generated)

### Conversation Entity
- `id`: UUID (required, immutable)
- `title`: String (optional)
- `user_id`: String (optional, for user identification)
- `created_at`: Timestamp (auto-generated)
- `updated_at`: Timestamp (auto-generated)

## Service Contracts

### OpenAI Service
- Must handle API rate limits gracefully
- Must provide meaningful error messages for API failures
- Must support both embedding generation and completion requests
- Must respect token limits and chunk large inputs appropriately

### Qdrant Vector Store Service
- Must maintain vector index consistency
- Must support semantic search with configurable thresholds
- Must handle batch operations efficiently
- Must provide reliable connectivity with proper retry mechanisms

### Database Service
- Must maintain ACID properties for all transactions
- Must support concurrent access safely
- Must provide proper connection pooling
- Must handle migrations gracefully

## Frontend Contracts

### Component Interfaces
- ChatInterface: Accepts `apiUrl` prop and provides chat functionality
- SelectionHandler: Provides selected text context to other components
- BookContentDisplay: Renders content with proper formatting and interactivity

### API Client Contract
- Must handle network errors gracefully
- Must provide loading states for all async operations
- Must implement proper request/response validation
- Must support authentication headers when required

## Error Handling Contracts

### Backend Error Responses
- Must return appropriate HTTP status codes
- Must provide meaningful error messages
- Must log errors for debugging purposes
- Must not expose internal implementation details

### Frontend Error Handling
- Must display user-friendly error messages
- Must provide fallback behaviors
- Must log errors for monitoring
- Must maintain application state integrity

## Performance Contracts

### Response Time Requirements
- API endpoints: < 2 seconds for typical requests
- Chat responses: < 5 seconds for complex queries
- Page load times: < 3 seconds for initial render
- Search operations: < 1 second for typical queries

### Scalability Requirements
- Must support 100+ concurrent users
- Must handle content corpus of 1000+ pages
- Must maintain performance as content grows
- Must support horizontal scaling