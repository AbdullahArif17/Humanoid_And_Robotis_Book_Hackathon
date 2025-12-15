# Constitution for "AI-Native Book + RAG Chatbot on Physical AI & Humanoid Robotics"

## Purpose
This constitution defines the strict rules governing all Spec-Kit Plus operations for this project. The assistant must follow these principles when generating specifications, plans, code, documentation, and book content.

## Core Principles
1. **Strict Adherence**: All implementations must follow the defined architecture and specifications exactly.
2. **Security First**: All API keys and sensitive information must be handled securely using environment variables.
3. **Modular Design**: Components must be loosely coupled and highly cohesive.
4. **Production Ready**: All code must be production-quality with proper error handling and validation.
5. **Documentation**: Every component must be thoroughly documented with clear explanations.

## Technical Constraints
- Backend must use FastAPI with async support
- Database must use PostgreSQL with SQLAlchemy ORM
- Vector store must use Qdrant Cloud for semantic search
- AI must use OpenAI API (GPT-4 or newer) for completions and embeddings
- Frontend must use Docusaurus v3 with React components
- All secrets must be stored in environment variables, never in code
- All API endpoints must include proper validation and error handling

## Quality Standards
- All code must follow established patterns and conventions
- Error handling must be comprehensive at all levels
- Logging must be implemented with appropriate levels
- Security practices must be followed for all user inputs
- Performance considerations must be taken into account
- Scalability must be planned for from the beginning

## Implementation Rules
- No hardcoded values except for development defaults
- All external service integrations must be configurable
- All components must be testable in isolation
- All user inputs must be validated before processing
- All API responses must include appropriate metadata
- All database operations must use proper transactions

## Review and Approval
Any deviation from this constitution requires explicit approval and must be documented.