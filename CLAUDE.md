# Claude Project File for "AI-Native Book + RAG Chatbot on Physical AI & Humanoid Robotics"

## Project Overview
This project implements an AI-Native Book with an integrated Retrieval-Augmented Generation (RAG) Chatbot focused on Physical AI & Humanoid Robotics. The system combines a Docusaurus-based technical book with an AI chatbot that can answer questions about the book content, including selected text queries.

## Project Structure
- `backend/`: FastAPI backend with PostgreSQL, Qdrant vector store, and OpenAI integration
- `book/`: Docusaurus v3 frontend with educational content organized in 4 modules
- `book/docs/`: Contains the 4 curriculum modules (ROS 2, Gazebo/Unity, NVIDIA Isaac, Vision-Language-Action)
- `backend/src/`: Backend source code with services, models, API endpoints, and AI components

## Key Features
- Educational content organized in 4 comprehensive modules
- RAG-based chatbot with semantic search capabilities
- Text selection integration for contextual queries
- Source citation in AI responses
- Conversation management with history tracking

## Technology Stack
- Backend: FastAPI, Python 3.11, SQLAlchemy, PostgreSQL (Neon Serverless)
- Vector Store: Qdrant Cloud for semantic search
- AI: OpenAI API (GPT-4 or newer) for completions and embeddings
- Frontend: Docusaurus v3, React, JavaScript
- Architecture: Microservices with clear separation of concerns

## Implementation Status
- [x] Project constitution and governance rules established
- [x] Full project specification created with user stories
- [x] Clarifications identified and incorporated into plan
- [x] Comprehensive project plan with architecture and data model
- [x] Implementation tasks defined and prioritized
- [x] Backend infrastructure implemented (database, config, vector store, models)
- [x] Frontend components created (chat interface, selection handler, API client)
- [x] API endpoints implemented with proper validation
- [x] Content ingestion pipeline created
- [ ] RAG functionality fully integrated
- [ ] Complete testing and validation
- [ ] Production deployment

## Next Steps
1. Complete RAG integration connecting frontend chat to backend AI services
2. Implement content ingestion pipeline to populate vector store
3. Test end-to-end functionality with sample queries
4. Deploy backend to cloud platform (Railway/Render/Fly.io)
5. Deploy frontend to GitHub Pages
6. Document deployment and usage procedures

## Special Considerations
- Security: All API keys and sensitive data must be properly managed via environment variables
- Scalability: System should handle concurrent users and large content corpus
- Accuracy: AI responses must be properly grounded in book content with citations
- Usability: Text selection and chat interface should be intuitive and responsive