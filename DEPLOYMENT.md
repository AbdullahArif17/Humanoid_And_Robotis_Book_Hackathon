# Deployment Guide

## Frontend (Docusaurus Book) - GitHub Pages

### Prerequisites
- GitHub account
- Git installed locally

### Steps
1. Fork this repository to your GitHub account
2. Clone your forked repository:
   ```bash
   git clone https://github.com/your-username/Humanoid_And_Robotis_Book_Hackathon.git
   cd Humanoid_And_Robotis_Book_Hackathon
   ```

3. Navigate to the book directory and install dependencies:
   ```bash
   cd book
   npm install
   ```

4. Build the static site:
   ```bash
   npm run build
   ```

5. Push changes to GitHub:
   ```bash
   git add .
   git commit -m "Update book content"
   git push origin main
   ```

6. Enable GitHub Pages:
   - Go to your repository on GitHub
   - Navigate to Settings > Pages
   - Select "Deploy from a branch"
   - Choose "main" branch and "/root" folder
   - Click "Save"

7. Your book will be available at: `https://your-username.github.io/Humanoid_And_Robotis_Book_Hackathon/`

## Backend (FastAPI API) - Railway

### Prerequisites
- Railway account (https://railway.app)
- OpenAI API key
- Qdrant Cloud account and API key
- Neon Postgres account and connection string

### Steps

1. Create a new Railway project
2. Connect to this GitHub repository
3. Add environment variables in Railway:
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `QDRANT_URL` - Your Qdrant Cloud URL
   - `QDRANT_API_KEY` - Your Qdrant API key
   - `DATABASE_URL` - Your Neon Postgres connection string
   - `SECRET_KEY` - A secure random secret key
   - `ALLOWED_ORIGINS` - `https://your-username.github.io` (replace with your GitHub Pages URL)

4. Deploy the project

5. Your API will be available at the Railway-generated URL

## Environment Variables

### Backend (.env)
```env
# Database Configuration
DATABASE_URL=postgresql://username:password@your-neon-db-url

# Qdrant Configuration
QDRANT_URL=https://your-qdrant-cluster.qdrant.tech
QDRANT_API_KEY=your-qdrant-api-key

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4-turbo

# Application Configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000
SECRET_KEY=your-production-secret-key

# API settings
ALLOWED_ORIGINS=https://your-username.github.io

# Performance settings
RESPONSE_MAX_TOKENS=1000
EMBEDDING_CHUNK_SIZE=1000
```

### Frontend (book/docusaurus.config.js)
Update these values in `docusaurus.config.js`:
- `url`: Your GitHub Pages URL (`https://your-username.github.io`)
- `baseUrl`: `/Humanoid_And_Robotis_Book_Hackathon/` (or `/` if using custom domain)
- `organizationName`: Your GitHub username
- `projectName`: `Humanoid_And_Robotis_Book_Hackathon`
- Update all GitHub links to point to your repository

## Content Ingestion

Before the RAG system works, you need to ingest the book content:

1. After deploying the backend, run the ingestion script:
   ```bash
   cd backend
   python -m src.scripts.run_ingestion ../../../book
   ```

2. This will parse all markdown files in the book/docs directory and store embeddings in Qdrant

## Health Check

Once deployed, verify your API is working:
- GET `https://your-api-url/health` - Should return: `{"status": "healthy", "service": "AI-Native Book RAG Chatbot API"}`

## Troubleshooting

### Frontend Issues
- If the chat interface doesn't connect to the backend, check the `apiUrl` prop in the `ChatInterface` component
- Ensure CORS settings in the backend allow your frontend domain

### Backend Issues
- Check that all environment variables are properly set
- Verify database and Qdrant connections
- Ensure OpenAI API key is valid

### Common Issues
- Make sure the book content has been ingested into the vector database
- Check that the frontend is pointing to the correct backend API URL
- Verify that both frontend and backend are deployed and accessible