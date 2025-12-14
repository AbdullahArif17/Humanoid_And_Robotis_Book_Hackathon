# API Documentation for AI-Native Book RAG Chatbot

## Overview

The AI-Native Book RAG Chatbot provides a REST API for managing educational content and chat interactions. The API follows REST principles and returns JSON responses.

## Base URL

```
http://localhost:8000/api/v1/
```

In production, replace `localhost:8000` with your actual domain.

## Authentication

Currently, the API does not require authentication. In a production environment, authentication would be implemented using API keys or JWT tokens.

## Content API (`/content/`)

### Create Content
- **Endpoint**: `POST /content/`
- **Description**: Creates a new book content entry
- **Parameters**:
  - `title` (string, required): Content title
  - `module_id` (string, required): Module identifier
  - `section_path` (string, required): Path to the section (e.g., "module-1-ros2/basics/nodes")
  - `content_type` (string, required): Type of content (text, code, diagram, exercise, lab)
  - `content_body` (string, required): Main content body
  - `metadata` (object, optional): Additional metadata (tags, difficulty, prerequisites)
  - `version` (integer, optional, default: 1): Content version
- **Response**: Created `BookContent` object
- **Status Codes**:
  - 200: Success
  - 400: Validation error
  - 500: Server error

### Get Content by ID
- **Endpoint**: `GET /content/{content_id}`
- **Description**: Retrieves content by its ID
- **Parameters**:
  - `content_id` (string): Unique identifier of the content
- **Response**: `BookContent` object
- **Status Codes**:
  - 200: Success
  - 404: Content not found
  - 500: Server error

### Get Content by Section Path
- **Endpoint**: `GET /content/path/{section_path}`
- **Description**: Retrieves content by its section path
- **Parameters**:
  - `section_path` (string): Path to the section (e.g., "module-1-ros2/basics/nodes")
- **Response**: `BookContent` object
- **Status Codes**:
  - 200: Success
  - 404: Content not found
  - 500: Server error

### Get Content by Module
- **Endpoint**: `GET /content/`
- **Description**: Retrieves content by module, optionally filtered by content type
- **Parameters**:
  - `module_id` (string, optional): Module identifier to filter by
  - `content_type` (string, optional): Content type to filter by
  - `skip` (integer, optional, default: 0): Number of records to skip
  - `limit` (integer, optional, default: 100, max: 1000): Maximum number of records to return
- **Response**: Array of `BookContent` objects
- **Status Codes**:
  - 200: Success
  - 500: Server error

### Update Content
- **Endpoint**: `PUT /content/{content_id}`
- **Description**: Updates an existing content entry
- **Parameters**:
  - `content_id` (string): Unique identifier of the content to update
  - `title` (string, optional): New title
  - `module_id` (string, optional): New module ID
  - `section_path` (string, optional): New section path
  - `content_type` (string, optional): New content type
  - `content_body` (string, optional): New content body
  - `metadata` (object, optional): New metadata
  - `version` (integer, optional): New version
- **Response**: Updated `BookContent` object
- **Status Codes**:
  - 200: Success
  - 400: Validation error
  - 404: Content not found
  - 500: Server error

### Delete Content
- **Endpoint**: `DELETE /content/{content_id}`
- **Description**: Deletes a content entry by its ID
- **Parameters**:
  - `content_id` (string): Unique identifier of the content to delete
- **Response**: Success message
- **Status Codes**:
  - 200: Success
  - 404: Content not found
  - 500: Server error

### Search Content
- **Endpoint**: `GET /content/search/`
- **Description**: Searches for content by title or content body
- **Parameters**:
  - `query` (string, required): Search query string (min length: 1)
  - `module_id` (string, optional): Optional module ID to limit search
- **Response**: Array of `BookContent` objects
- **Status Codes**:
  - 200: Success
  - 500: Server error

## Chat API (`/chat/`)

### Create Conversation
- **Endpoint**: `POST /chat/conversations/`
- **Description**: Creates a new conversation
- **Parameters**:
  - `user_id` (string, optional): Optional user identifier
  - `title` (string, optional): Optional conversation title
- **Response**: Created `Conversation` object
- **Status Codes**:
  - 200: Success
  - 500: Server error

### Get Conversation
- **Endpoint**: `GET /chat/conversations/{conversation_id}`
- **Description**: Retrieves a conversation by its ID
- **Parameters**:
  - `conversation_id` (string): Unique identifier of the conversation
- **Response**: `Conversation` object
- **Status Codes**:
  - 200: Success
  - 404: Conversation not found
  - 500: Server error

### Get Conversations by User
- **Endpoint**: `GET /chat/conversations/`
- **Description**: Retrieves conversations for a specific user with pagination
- **Parameters**:
  - `user_id` (string, required): User identifier
  - `skip` (integer, optional, default: 0): Number of records to skip
  - `limit` (integer, optional, default: 20, max: 100): Maximum number of records to return
- **Response**: Array of `Conversation` objects
- **Status Codes**:
  - 200: Success
  - 500: Server error

### Process Query
- **Endpoint**: `POST /chat/conversations/{conversation_id}/query`
- **Description**: Processes a user query and generates an AI response using RAG
- **Parameters**:
  - `conversation_id` (string): Unique identifier of the conversation
  - `query_text` (string, required): User's query text (min length: 1)
  - `selected_text` (string, optional): Optional selected text for context
  - `context_window` (integer, optional, default: 5, range: 1-20): Number of surrounding chunks to include
- **Response**: Generated `ChatbotResponse` object
- **Status Codes**:
  - 200: Success
  - 400: Validation error
  - 404: Conversation not found
  - 500: Server error

### Get Chat History
- **Endpoint**: `GET /chat/conversations/{conversation_id}/history`
- **Description**: Retrieves chat history for a conversation
- **Parameters**:
  - `conversation_id` (string): Unique identifier of the conversation
  - `limit` (integer, optional, default: 50, max: 200): Maximum number of messages to return
- **Response**: Array of chat messages with user queries and AI responses
- **Status Codes**:
  - 200: Success
  - 500: Server error

### Update Conversation Title
- **Endpoint**: `PUT /chat/conversations/{conversation_id}/title`
- **Description**: Updates the title of a conversation
- **Parameters**:
  - `conversation_id` (string): Unique identifier of the conversation
  - `title` (string, required): New title for the conversation (min length: 1, max: 200)
- **Response**: Success message
- **Status Codes**:
  - 200: Success
  - 400: Validation error
  - 404: Conversation not found
  - 500: Server error

### Delete Conversation
- **Endpoint**: `DELETE /chat/conversations/{conversation_id}`
- **Description**: Deletes a conversation and all associated queries and responses
- **Parameters**:
  - `conversation_id` (string): Unique identifier of the conversation to delete
- **Response**: Success message
- **Status Codes**:
  - 200: Success
  - 404: Conversation not found
  - 500: Server error

## Error Handling

The API returns appropriate HTTP status codes and JSON error responses:

```json
{
  "detail": "Error message"
}
```

Common status codes:
- 200: Success
- 400: Bad request (validation errors)
- 404: Resource not found
- 500: Internal server error

## Environment Variables

The application uses the following environment variables:

- `DATABASE_URL`: Database connection string (default: "postgresql://user:password@localhost/book_chatbot")
- `QDRANT_URL`: Qdrant Cloud URL (default: "https://your-cluster.qdrant.tech")
- `QDRANT_API_KEY`: Qdrant API key (default: "your-api-key")
- `OPENAI_API_KEY`: OpenAI API key (required)
- `OPENAI_MODEL`: OpenAI model to use (default: "gpt-4")
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model (default: "text-embedding-ada-002")
- `HOST`: Host to bind to (default: "127.0.0.1")
- `PORT`: Port to bind to (default: 8000)
- `DEBUG`: Enable debug mode (default: false)
- `LOG_LEVEL`: Logging level (default: "INFO")
- `QDRANT_COLLECTION_NAME`: Qdrant collection name (default: "book_content")
- `CORS_ORIGINS`: Comma-separated list of allowed origins (default: ["*"])
- `OPENAI_TIMEOUT`: Timeout for OpenAI requests in seconds (default: 30)
- `OPENAI_MAX_RETRIES`: Maximum retries for OpenAI requests (default: 3)
- `OPENAI_BASE_URL`: Base URL for OpenAI API (optional, for custom endpoints)