---
title: AI-Native Book RAG Chatbot Backend
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "3.11"
app_file: app.py
pinned: false
---

# AI-Native Book RAG Chatbot Backend

This is the backend API for the AI-Native Book on Physical AI & Humanoid Robotics. It provides a RAG (Retrieval-Augmented Generation) chatbot that can answer questions about the book content.

## Hugging Face Spaces Deployment

This backend can be deployed on Hugging Face Spaces using the following configuration:

### Files Required:
- `app.py` - Entry point for the FastAPI application
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container configuration
- `space.yml` - Hugging Face Spaces configuration

### Environment Variables:
The following environment variables need to be set in the Space settings:

**Secrets (add as "Secrets" for security):**
- `OPENAI_API_KEY` (required) - Your OpenAI API key
- `QDRANT_API_KEY` (required) - Your Qdrant API key
- `DATABASE_URL` - Database connection string (if contains credentials)
- `SECRET_KEY` - Secret key for security

**Environment Variables (add as "Variables"):**
- `QDRANT_URL` (required) - Your Qdrant Cloud URL
- `ALLOWED_ORIGINS` - Comma-separated list of allowed origins (set to your frontend URL)
- `DEBUG` - Set to "true" for debugging (defaults to "false")

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