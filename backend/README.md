# AI-Native Book RAG Chatbot - Backend

This is the backend service for the AI-Native Book RAG Chatbot application focused on Physical AI & Humanoid Robotics. The backend provides a FastAPI-based REST API for managing educational content and chat interactions using Retrieval-Augmented Generation (RAG).

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)

## Features

- **Content Management**: Create, read, update, and delete educational content organized into modules and sections
- **RAG-Based Chat**: AI-powered chatbot that answers questions based on the educational content
- **Semantic Search**: Vector-based semantic search using Qdrant for relevant content retrieval
- **Conversation Management**: Track and manage chat conversations with history
- **Source Citations**: AI responses include citations to the source content
- **RESTful API**: Well-documented API with OpenAPI/Swagger support

## Architecture

The backend follows a service-oriented architecture with the following layers:

- **API Layer**: FastAPI endpoints for external communication
- **Service Layer**: Business logic and data validation
- **Data Layer**: SQLAlchemy ORM for database operations
- **Vector Store**: Qdrant for semantic search and content retrieval
- **AI Layer**: OpenAI API for content generation and embeddings

## Prerequisites

- Python 3.11+
- PostgreSQL database (or compatible)
- Qdrant Cloud account or self-hosted Qdrant instance
- OpenAI API key

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the backend directory with the required environment variables:

   ```env
   DATABASE_URL=postgresql://user:password@localhost/book_chatbot
   QDRANT_URL=https://your-qdrant-cluster.qdrant.tech
   QDRANT_API_KEY=your-qdrant-api-key
   OPENAI_API_KEY=your-openai-api-key
   OPENAI_MODEL=gpt-4
   DEBUG=true
   ```

2. For a complete list of environment variables, see the [Environment Variables](#environment-variables) section.

## Running the Application

### Development

```bash
cd backend
python -m src.main
```

Or using uvicorn directly:

```bash
cd backend
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
```

### Production

```bash
cd backend
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Documentation

The API provides automatic OpenAPI documentation at:
- `http://localhost:8000/docs` - Swagger UI
- `http://localhost:8000/redoc` - ReDoc documentation
- `http://localhost:8000/openapi.json` - OpenAPI schema

For detailed API documentation, see [docs/api.md](docs/api.md).

## Testing

To run the tests:

```bash
cd backend
python -m pytest tests/
```

For detailed test coverage:

```bash
python -m pytest tests/ --cov=src --cov-report=html
```

## Docker Deployment

### Building and Running with Docker

1. Build the Docker image:
   ```bash
   docker build -t book-chatbot-backend .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env book-chatbot-backend
   ```

### Using Docker Compose

Use the provided `docker-compose.yml` file to run the complete application stack:

```bash
docker-compose up --build
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://user:password@localhost/book_chatbot` | Database connection string |
| `QDRANT_URL` | `https://your-cluster.qdrant.tech` | Qdrant Cloud URL |
| `QDRANT_API_KEY` | `your-api-key` | Qdrant API key |
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4` | OpenAI model to use for completions |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-ada-002` | OpenAI model to use for embeddings |
| `HOST` | `127.0.0.1` | Host to bind to |
| `PORT` | `8000` | Port to bind to |
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `QDRANT_COLLECTION_NAME` | `book_content` | Qdrant collection name |
| `CORS_ORIGINS` | `["*"]` | Comma-separated list of allowed origins |
| `OPENAI_TIMEOUT` | `30` | Timeout for OpenAI requests in seconds |
| `OPENAI_MAX_RETRIES` | `3` | Maximum retries for OpenAI requests |
| `OPENAI_BASE_URL` | - | Base URL for OpenAI API (optional, for custom endpoints) |

## Project Structure

```
backend/
├── src/                    # Source code
│   ├── main.py            # Main application entry point
│   ├── config.py          # Configuration settings
│   ├── database/          # Database configuration
│   ├── models/            # SQLAlchemy models
│   ├── services/          # Business logic services
│   ├── api/               # API endpoints
│   ├── ai/                # AI/ML components
│   ├── vector_store/      # Vector store integration
│   └── utils/             # Utility functions
├── tests/                 # Test files
├── docs/                  # Documentation
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── Dockerfile            # Docker configuration
├── pyproject.toml        # Project configuration (formatting, testing)
├── .env.example          # Example environment variables
└── README.md             # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`python -m pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Hugging Face Spaces Deployment

This backend can be deployed on Hugging Face Spaces using the following configuration:

### Files Required:
- `app.py` - Entry point for the FastAPI application
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `space.yml` - Hugging Face Spaces configuration

### Environment Variables:
The following environment variables need to be set in the Space settings:

- `OPENAI_API_KEY` (required) - Your OpenAI API key
- `QDRANT_URL` (required) - Your Qdrant Cloud URL
- `QDRANT_API_KEY` (required) - Your Qdrant API key
- `DATABASE_URL` - Database connection string (defaults to SQLite)
- `ALLOWED_ORIGINS` - Comma-separated list of allowed origins (set to your frontend URL)
- `DEBUG` - Set to "true" for debugging (defaults to "false")
- `SECRET_KEY` - Secret key for security (defaults to a placeholder)

### Setup Instructions:

1. Fork this repository to your GitHub account
2. Create a new Space on Hugging Face with the "Docker" SDK option
3. Point the Space to your forked repository
4. Add the required environment variables in the Space settings (in the "Files and Environment" section)
5. Wait for the Space to build and deploy

### API Endpoints:

- `GET /` - Root endpoint
- `GET /health` - Health check
- `POST /api/embed` - Embed content endpoint
- `POST /api/query` - Query the full book
- `POST /api/query-selected` - Query selected text

### Note:
Before the RAG system works, you need to ingest the book content by running the ingestion script once the backend is deployed.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.