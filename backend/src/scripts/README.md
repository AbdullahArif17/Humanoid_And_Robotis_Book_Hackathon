# Scripts

This directory contains utility scripts for the AI-Native Book RAG Chatbot application.

## Content Ingestion Script

The content ingestion script processes Docusaurus markdown files and populates the database and vector store.

### Usage

```bash
cd backend
python -m src.scripts.run_ingestion [path_to_book_directory]
```

If no path is provided, the script defaults to `../../../book`.

### What the script does

1. Recursively processes all markdown files in the book/docs directory
2. Extracts content information including title, module, section path, and metadata
3. Creates database entries for each content piece
4. Generates embeddings using OpenAI
5. Stores embeddings in Qdrant vector database

### Requirements

- Database must be running and accessible
- Qdrant vector database must be running and accessible
- OpenAI API key must be configured in environment variables