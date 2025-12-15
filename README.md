# AI-Native Book + RAG Chatbot: Physical AI & Humanoid Robotics

A comprehensive educational platform combining a Docusaurus-based book on Physical AI & Humanoid Robotics with an AI-powered chatbot that answers questions using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- **Educational Content**: 4 comprehensive modules on Physical AI & Humanoid Robotics
  - Module 1: ROS 2 for Humanoid Robotics
  - Module 2: Gazebo & Unity for Humanoid Simulation
  - Module 3: NVIDIA Isaac for Humanoid AI
  - Module 4: Vision-Language-Action for Humanoid Robotics

- **AI-Powered Chatbot**: Integrated RAG system that answers questions about book content
  - Full-book queries with semantic search
  - Selected-text context queries
  - Source citations with links to original content
  - Conversation history management

- **Modern Architecture**: FastAPI backend with Docusaurus frontend

## 📋 Prerequisites

- **Backend**: Python 3.11+, PostgreSQL, Qdrant Cloud, OpenAI API
- **Frontend**: Node.js 20+ (recommended), Node.js 18+ (minimum for development)
- **Deployment**: GitHub account, Railway account (or similar)

## 🛠️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Docusaurus    │◄──►│   FastAPI API    │◄──►│  PostgreSQL DB  │
│   Frontend      │    │   Backend        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Qdrant Vector   │
                    │     Database      │
                    └───────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   OpenAI API      │
                    │   (GPT-4)         │
                    └───────────────────┘
```

## 📦 Deployment

### Frontend (GitHub Pages)
1. Fork this repository
2. Update `book/docusaurus.config.js` with your GitHub username
3. Run `npm install && npm run build` in the `book` directory
4. Push to GitHub
5. Enable GitHub Pages in repository settings
6. Access at `https://your-username.github.io/Humanoid_And_Robotis_Book_Hackathon/`

### Backend (Railway)
1. Create Railway account
2. Connect to your forked repository
3. Add environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `DATABASE_URL`: Neon Postgres connection string
   - `QDRANT_URL`: Qdrant Cloud URL
   - `QDRANT_API_KEY`: Qdrant API key
   - `SECRET_KEY`: Secure random key
   - `ALLOWED_ORIGINS`: `https://your-username.github.io`
4. Deploy

### Content Ingestion
After backend deployment, run the ingestion script to populate the vector database:
```bash
cd backend
python -m src.scripts.run_ingestion ../../../book
```

## 🌐 API Endpoints

- `POST /api/query` - Full-book RAG queries
- `POST /api/query-selected` - Selected-text RAG queries
- `GET /api/health` - Health check
- `POST /api/embed` - Content embedding

## 🧪 Testing

- Health check: `GET /api/health`
- Frontend chat interface connects to backend API
- All book navigation works

## 📄 License

MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

**Note**: This project was developed for the Humanoid And Robotis Book Hackathon. It represents a complete solution for creating an AI-native educational platform focused on Physical AI & Humanoid Robotics.