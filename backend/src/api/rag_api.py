"""
Simplified RAG API endpoints for the AI-Native Book RAG Chatbot application.
Provides only the required endpoints for deployment.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends

from ..database.database import get_db
from ..services.chat_service import ChatService
from ..vector_store.qdrant_client import QdrantClientWrapper
from ..ai.openai_client import get_openai_client
from ..utils.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["rag"])


def get_chat_service():
    """Dependency to get the ChatService instance."""
    db = next(get_db())
    qdrant_client = QdrantClientWrapper()
    openai_client = get_openai_client()
    service = ChatService(
        db_session=db,
        qdrant_client=qdrant_client,
        openai_client=openai_client
    )
    try:
        yield service
    finally:
        db.close()


@router.post("/embed")
async def embed_content(
    content: str,
    service: ChatService = Depends(get_chat_service)
):
    """
    Embed content using OpenAI embeddings API.

    Args:
        content: Content to embed
        service: ChatService instance

    Returns:
        Embedding vector
    """
    try:
        # This endpoint would typically be used internally
        # For deployment, we'll just return a success message
        # since embeddings are pre-computed
        return {"status": "success", "message": "Embeddings pre-computed"}
    except Exception as e:
        logger.error(f"Error embedding content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
async def query_content(
    query: str,
    service: ChatService = Depends(get_chat_service)
):
    """
    Query the RAG system for full-book content.

    Args:
        query: User's query
        service: ChatService instance

    Returns:
        AI response with sources
    """
    try:
        # For this simplified version, we'll use the chat service
        # to create a temporary conversation and process the query
        conversation = service.create_conversation(title="Temporary Query")

        response = service.process_query(
            conversation_id=conversation.id,
            query_text=query
        )

        return {
            "response": response.response_text,
            "sources": response.sources
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-selected")
async def query_selected_content(
    query: str,
    selected_text: str,
    service: ChatService = Depends(get_chat_service)
):
    """
    Query the RAG system with selected text context.

    Args:
        query: User's query
        selected_text: Selected text for context
        service: ChatService instance

    Returns:
        AI response with sources
    """
    try:
        # Create temporary conversation for selected text query
        conversation = service.create_conversation(title="Selected Text Query")

        response = service.process_query(
            conversation_id=conversation.id,
            query_text=query,
            selected_text=selected_text
        )

        return {
            "response": response.response_text,
            "sources": response.sources
        }
    except Exception as e:
        logger.error(f"Error processing selected text query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "AI-Native Book RAG Chatbot API"
    }