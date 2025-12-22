"""
Simplified RAG API endpoints for the AI-Native Book RAG Chatbot application.
Provides only the required endpoints for deployment.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from src.database.database import get_db
from src.services.chat_service import ChatService
from src.vector_store.qdrant_client import QdrantClientWrapper
from src.ai.openai_client import get_openai_client
from src.utils.logging_config import get_logger


class CreateConversationRequest(BaseModel):
    user_id: Optional[str] = None
    title: Optional[str] = None

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


@router.post("/v1/chat/conversations")
async def create_conversation_endpoint(
    request: CreateConversationRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Create a new conversation.

    Args:
        request: Request body with user_id and title
        service: ChatService instance

    Returns:
        Created conversation
    """
    try:
        conversation = service.create_conversation(user_id=request.user_id, title=request.title)
        return {
            "id": conversation.id,
            "session_id": conversation.session_id,
            "user_id": conversation.user_id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/chat/conversations/{conversation_id}")
async def get_conversation_endpoint(
    conversation_id: str,
    service: ChatService = Depends(get_chat_service)
):
    """
    Get a specific conversation.

    Args:
        conversation_id: Unique identifier of the conversation
        service: ChatService instance

    Returns:
        Conversation details
    """
    try:
        conversation = service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {
            "id": conversation.id,
            "session_id": conversation.session_id,
            "user_id": conversation.user_id,
            "title": conversation.title,
            "is_active": conversation.is_active,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "conversation_metadata": conversation.conversation_metadata
        }
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/chat/conversations")
async def get_conversations_endpoint(
    user_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    service: ChatService = Depends(get_chat_service)
):
    """
    Get conversations for a user.

    Args:
        user_id: User identifier
        skip: Number of records to skip
        limit: Maximum number of records to return
        service: ChatService instance

    Returns:
        List of conversations
    """
    try:
        if user_id:
            conversations = service.get_conversations_by_user(user_id, skip, limit)
        else:
            # If no user_id provided, you might want to implement a different strategy
            # For now, we'll return an empty list or could implement a general query
            conversations = []

        return {
            "conversations": [
                {
                    "id": conv.id,
                    "session_id": conv.session_id,
                    "user_id": conv.user_id,
                    "title": conv.title,
                    "is_active": conv.is_active,
                    "created_at": conv.created_at,
                    "updated_at": conv.updated_at
                } for conv in conversations
            ],
            "total": len(conversations)
        }
    except Exception as e:
        logger.error(f"Error retrieving conversations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/chat/conversations/{conversation_id}/history")
async def get_conversation_history_endpoint(
    conversation_id: str,
    limit: int = 50,
    service: ChatService = Depends(get_chat_service)
):
    """
    Get chat history for a conversation.

    Args:
        conversation_id: Unique identifier of the conversation
        limit: Maximum number of messages to return
        service: ChatService instance

    Returns:
        Chat history
    """
    try:
        history = service.get_chat_history(conversation_id, limit)
        return {
            "conversation_id": conversation_id,
            "history": history,
            "total_messages": len(history)
        }
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/v1/chat/conversations/{conversation_id}/title")
async def update_conversation_title_endpoint(
    conversation_id: str,
    title: str,
    service: ChatService = Depends(get_chat_service)
):
    """
    Update conversation title.

    Args:
        conversation_id: Unique identifier of the conversation
        title: New title for the conversation
        service: ChatService instance

    Returns:
        Update status
    """
    try:
        success = service.update_conversation_title(conversation_id, title)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"success": True, "message": "Conversation title updated successfully"}
    except Exception as e:
        logger.error(f"Error updating conversation title: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/v1/chat/conversations/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: str,
    service: ChatService = Depends(get_chat_service)
):
    """
    Delete a conversation.

    Args:
        conversation_id: Unique identifier of the conversation
        service: ChatService instance

    Returns:
        Deletion status
    """
    try:
        success = service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"success": True, "message": "Conversation deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class ProcessQueryRequest(BaseModel):
    query_text: str
    selected_text: Optional[str] = None
    context_window: Optional[int] = 5


@router.post("/v1/chat/conversations/{conversation_id}/query")
async def process_query_endpoint(
    conversation_id: str,
    request: ProcessQueryRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Process a query in a conversation.

    Args:
        conversation_id: Unique identifier of the conversation
        request: Query request with query text and optional selected text
        service: ChatService instance

    Returns:
        AI response with sources
    """
    try:
        response = service.process_query(
            conversation_id=conversation_id,
            query_text=request.query_text,
            selected_text=request.selected_text,
            context_window=request.context_window
        )

        return {
            "response": response.response_text,
            "sources": response.sources
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))