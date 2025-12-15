# Quickstart Guide: AI-Native Book + RAG Chatbot

## Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL-compatible database
- Qdrant Cloud account
- OpenAI API key

## Setup Instructions

### Backend Setup
1. Navigate to the backend directory: `cd backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`:
   ```
   OPENAI_API_KEY=your-openai-api-key
   DATABASE_URL=postgresql://user:password@localhost/dbname
   QDRANT_URL=https://your-cluster.qdrant.tech
   QDRANT_API_KEY=your-api-key
   ```
4. Run the application: `python -m src.main`

### Frontend Setup
1. Navigate to the book directory: `cd book`
2. Install dependencies: `npm install`
3. Start development server: `npm start`

### Content Ingestion
1. Place book content in the `book/docs/` directory
2. Run the ingestion script: `python -m src.scripts.run_ingestion ../../../book`
3. Verify content is indexed in Qdrant

## Basic Usage
- Backend API available at `http://localhost:8000`
- Frontend available at `http://localhost:3000`
- API documentation available at `http://localhost:8000/docs`

## Common Issues
- Ensure all environment variables are properly set
- Verify database and Qdrant connections
- Check that OpenAI API key is valid and has sufficient quota
- Confirm that content has been properly ingested into the vector store