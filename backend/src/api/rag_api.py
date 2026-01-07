"""
Simplified RAG API endpoints for the AI-Native Book RAG Chatbot application.
Provides only the required endpoints for deployment.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status
from src.database.database import get_db
from src.services.chat_service import ChatService
from src.vector_store.qdrant_client import QdrantClientWrapper
from src.ai.google_client import get_google_client
from src.utils.logging_config import get_logger
from src.utils.exceptions import ValidationError, DatabaseError, RAGError, AIClientError
from src.api.schemas import (
    CreateConversationRequest,
    ProcessQueryRequest,
    EmbedContentRequest,
    QueryRequest,
    QuerySelectedRequest,
    UpdateTitleRequest,
    ConversationResponse,
    ConversationListResponse,
    QueryResponse,
    ChatHistoryResponse,
    HealthResponse,
    SuccessResponse,
    ErrorResponse
)

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["rag"])


def get_chat_service():
    """Dependency to get the ChatService instance."""
    db = next(get_db())
    qdrant_client = QdrantClientWrapper()
    google_client = get_google_client()
    service = ChatService(
        db_session=db,
        qdrant_client=qdrant_client,
        google_client=google_client
    )
    try:
        yield service
    finally:
        db.close()


@router.post("/embed", response_model=SuccessResponse)
async def embed_content(
    request: EmbedContentRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Embed content using OpenAI embeddings API.

    **Note**: This endpoint returns a success message since embeddings are pre-computed.
    
    Args:
        request: EmbedContentRequest with content to embed
        service: ChatService instance

    Returns:
        SuccessResponse indicating embeddings are pre-computed
    """
    try:
        # This endpoint would typically be used internally
        # For deployment, we'll just return a success message
        # since embeddings are pre-computed
        return SuccessResponse(message="Embeddings pre-computed")
    except ValidationError as e:
        logger.warning(f"Validation error embedding content: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error embedding content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing embedding request"
        )


@router.post("/query", response_model=QueryResponse)
async def query_content(
    request: QueryRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Query the RAG system for full-book content.

    **Example**:
    ```json
    {
        "query": "What is ROS 2?"
    }
    ```

    Args:
        request: QueryRequest with user's query
        service: ChatService instance

    Returns:
        QueryResponse with AI response and sources
    """
    try:
        # For this simplified version, we'll use the chat service
        # to create a temporary conversation and process the query
        conversation = service.create_conversation(title="Temporary Query")

        response = await service.process_query(
            conversation_id=conversation.id,
            query_text=request.query
        )

        return QueryResponse(
            response=response.response_text,
            sources=response.sources
        )
    except ValidationError as e:
        logger.warning(f"Validation error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RAGError as e:
        logger.error(f"RAG error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving relevant content"
        )
    except AIClientError as e:
        logger.error(f"AI client error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing query"
        )


@router.post("/query-selected", response_model=QueryResponse)
async def query_selected_content(
    request: QuerySelectedRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Query the RAG system with selected text context.

    **Example**:
    ```json
    {
        "query": "Explain this concept",
        "selected_text": "ROS 2 is a middleware framework for robotics..."
    }
    ```

    Args:
        request: QuerySelectedRequest with query and selected text
        service: ChatService instance

    Returns:
        QueryResponse with AI response and sources
    """
    try:
        # Create temporary conversation for selected text query
        conversation = service.create_conversation(title="Selected Text Query")

        response = await service.process_query(
            conversation_id=conversation.id,
            query_text=request.query,
            selected_text=request.selected_text
        )

        return QueryResponse(
            response=response.response_text,
            sources=response.sources
        )
    except ValidationError as e:
        logger.warning(f"Validation error processing selected text query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RAGError as e:
        logger.error(f"RAG error processing selected text query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving relevant content"
        )
    except AIClientError as e:
        logger.error(f"AI client error processing selected text query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Error processing selected text query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing query"
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint with dependency status.

    Returns:
        HealthResponse with service status and dependency health
    """
    dependencies = {}
    
    # Check database connection
    try:
        from src.database.database import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        dependencies["database"] = "healthy"
    except Exception as e:
        logger.warning(f"Database health check failed: {str(e)}")
        dependencies["database"] = "unhealthy"
    
    # Check Qdrant connection
    try:
        qdrant_client = QdrantClientWrapper()
        # Try to get collections
        qdrant_client.client.get_collections()
        dependencies["qdrant"] = "healthy"
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {str(e)}")
        dependencies["qdrant"] = "unhealthy"
    
    # Determine overall health
    all_healthy = all(status == "healthy" for status in dependencies.values())
    status = "healthy" if all_healthy else "degraded"
    
    return HealthResponse(
        status=status,
        service="AI-Native Book RAG Chatbot API",
        version="1.0.0",
        dependencies=dependencies
    )


@router.post("/v1/chat/conversations", response_model=ConversationResponse)
@router.post("/v1/chat/conversations/", response_model=ConversationResponse)  # Explicitly support trailing slash
async def create_conversation_endpoint(
    request: CreateConversationRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Create a new conversation.

    **Example**:
    ```json
    {
        "user_id": "user-123",
        "title": "My Conversation"
    }
    ```

    Args:
        request: CreateConversationRequest with user_id and title
        service: ChatService instance

    Returns:
        ConversationResponse with created conversation details
    """
    try:
        conversation = service.create_conversation(user_id=request.user_id, title=request.title)
        return ConversationResponse(
            id=conversation.id,
            session_id=conversation.session_id,
            user_id=conversation.user_id,
            title=conversation.title,
            is_active=conversation.is_active,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            conversation_metadata=conversation.conversation_metadata
        )
    except ValidationError as e:
        logger.warning(f"Validation error creating conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except DatabaseError as e:
        logger.error(f"Database error creating conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating conversation"
        )
    except Exception as e:
        logger.error(f"Error creating conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while creating conversation"
        )


@router.get("/v1/chat/conversations/{conversation_id}", response_model=ConversationResponse)
@router.get("/v1/chat/conversations/{conversation_id}/", response_model=ConversationResponse)  # Support trailing slash
async def get_conversation_endpoint(
    conversation_id: str,
    service: ChatService = Depends(get_chat_service)
):
    """
    Get a specific conversation by ID.

    Args:
        conversation_id: Unique identifier of the conversation (UUID)
        service: ChatService instance

    Returns:
        ConversationResponse with conversation details
    """
    try:
        from src.utils.validators import validate_conversation_id
        validation_errors = validate_conversation_id(conversation_id)
        if validation_errors:
            raise ValidationError(f"Invalid conversation ID: {', '.join(validation_errors)}")
        
        conversation = service.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        return ConversationResponse(
            id=conversation.id,
            session_id=conversation.session_id,
            user_id=conversation.user_id,
            title=conversation.title,
            is_active=conversation.is_active,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            conversation_metadata=conversation.conversation_metadata
        )
    except ValidationError as e:
        logger.warning(f"Validation error retrieving conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving conversation"
        )


@router.get("/v1/chat/conversations", response_model=ConversationListResponse)
@router.get("/v1/chat/conversations/", response_model=ConversationListResponse)  # Support trailing slash
async def get_conversations_endpoint(
    user_id: Optional[str] = None,
    skip: int = 0,
    limit: int = 20,
    service: ChatService = Depends(get_chat_service)
):
    """
    Get conversations for a user with pagination.

    Args:
        user_id: Optional user identifier
        skip: Number of records to skip (default: 0)
        limit: Maximum number of records to return (default: 20, max: 100)
        service: ChatService instance

    Returns:
        ConversationListResponse with list of conversations
    """
    try:
        from src.utils.validators import validate_pagination_params, validate_user_id
        
        # Validate pagination
        pagination_errors = validate_pagination_params(skip, limit, max_limit=100)
        if pagination_errors:
            raise ValidationError(f"Invalid pagination parameters: {', '.join(pagination_errors)}")
        
        # Validate user_id if provided
        user_id_errors = validate_user_id(user_id)
        if user_id_errors:
            raise ValidationError(f"Invalid user ID: {', '.join(user_id_errors)}")
        
        if user_id:
            conversations = service.get_conversations_by_user(user_id, skip, limit)
        else:
            # If no user_id provided, return empty list
            conversations = []

        return ConversationListResponse(
            conversations=[
                ConversationResponse(
                    id=conv.id,
                    session_id=conv.session_id,
                    user_id=conv.user_id,
                    title=conv.title,
                    is_active=conv.is_active,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    conversation_metadata=conv.conversation_metadata
                ) for conv in conversations
            ],
            total=len(conversations),
            skip=skip,
            limit=limit
        )
    except ValidationError as e:
        logger.warning(f"Validation error retrieving conversations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving conversations: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving conversations"
        )


@router.get("/v1/chat/conversations/{conversation_id}/history", response_model=ChatHistoryResponse)
@router.get("/v1/chat/conversations/{conversation_id}/history/", response_model=ChatHistoryResponse)  # Support trailing slash
async def get_conversation_history_endpoint(
    conversation_id: str,
    limit: int = 50,
    service: ChatService = Depends(get_chat_service)
):
    """
    Get chat history for a conversation.

    Args:
        conversation_id: Unique identifier of the conversation (UUID)
        limit: Maximum number of messages to return (default: 50, max: 100)
        service: ChatService instance

    Returns:
        ChatHistoryResponse with conversation history
    """
    try:
        from src.utils.validators import validate_conversation_id, validate_pagination_params
        
        # Validate conversation_id
        conv_errors = validate_conversation_id(conversation_id)
        if conv_errors:
            raise ValidationError(f"Invalid conversation ID: {', '.join(conv_errors)}")
        
        # Validate limit
        pagination_errors = validate_pagination_params(0, limit, max_limit=100)
        if pagination_errors:
            raise ValidationError(f"Invalid limit: {', '.join(pagination_errors)}")
        
        history = service.get_chat_history(conversation_id, limit)
        return ChatHistoryResponse(
            conversation_id=conversation_id,
            history=history,
            total_messages=len(history)
        )
    except ValidationError as e:
        logger.warning(f"Validation error retrieving chat history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving chat history"
        )


@router.put("/v1/chat/conversations/{conversation_id}/title", response_model=SuccessResponse)
@router.put("/v1/chat/conversations/{conversation_id}/title/", response_model=SuccessResponse)  # Support trailing slash
async def update_conversation_title_endpoint(
    conversation_id: str,
    request: UpdateTitleRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Update conversation title.

    Args:
        conversation_id: Unique identifier of the conversation (UUID)
        request: UpdateTitleRequest with new title
        service: ChatService instance

    Returns:
        SuccessResponse indicating success
    """
    try:
        from src.utils.validators import validate_conversation_id
        
        # Validate conversation_id
        conv_errors = validate_conversation_id(conversation_id)
        if conv_errors:
            raise ValidationError(f"Invalid conversation ID: {', '.join(conv_errors)}")
        
        success = service.update_conversation_title(conversation_id, request.title)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        return SuccessResponse(message="Conversation title updated successfully")
    except ValidationError as e:
        logger.warning(f"Validation error updating conversation title: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation title: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while updating conversation title"
        )


@router.delete("/v1/chat/conversations/{conversation_id}", response_model=SuccessResponse)
@router.delete("/v1/chat/conversations/{conversation_id}/", response_model=SuccessResponse)  # Support trailing slash
async def delete_conversation_endpoint(
    conversation_id: str,
    service: ChatService = Depends(get_chat_service)
):
    """
    Delete a conversation.

    Args:
        conversation_id: Unique identifier of the conversation (UUID)
        service: ChatService instance

    Returns:
        SuccessResponse indicating success
    """
    try:
        from src.utils.validators import validate_conversation_id
        
        # Validate conversation_id
        conv_errors = validate_conversation_id(conversation_id)
        if conv_errors:
            raise ValidationError(f"Invalid conversation ID: {', '.join(conv_errors)}")
        
        success = service.delete_conversation(conversation_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        return SuccessResponse(message="Conversation deleted successfully")
    except ValidationError as e:
        logger.warning(f"Validation error deleting conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while deleting conversation"
        )


@router.post("/v1/chat/conversations/{conversation_id}/query", response_model=QueryResponse)
@router.post("/v1/chat/conversations/{conversation_id}/query/", response_model=QueryResponse)  # Support trailing slash
async def process_query_endpoint(
    conversation_id: str,
    request: ProcessQueryRequest,
    service: ChatService = Depends(get_chat_service)
):
    """
    Process a query in a conversation.

    **Example**:
    ```json
    {
        "query_text": "What is ROS 2?",
        "selected_text": "ROS 2 is a middleware framework...",
        "context_window": 5
    }
    ```

    Args:
        conversation_id: Unique identifier of the conversation (UUID)
        request: ProcessQueryRequest with query text and optional selected text
        service: ChatService instance

    Returns:
        QueryResponse with AI response and sources
    """
    try:
        from src.utils.validators import validate_conversation_id
        
        # Validate conversation_id
        conv_errors = validate_conversation_id(conversation_id)
        if conv_errors:
            raise ValidationError(f"Invalid conversation ID: {', '.join(conv_errors)}")
        
        response = await service.process_query(
            conversation_id=conversation_id,
            query_text=request.query_text,
            selected_text=request.selected_text,
            context_window=request.context_window
        )

        return QueryResponse(
            response=response.response_text,
            sources=response.sources
        )
    except ValidationError as e:
        logger.warning(f"Validation error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RAGError as e:
        logger.error(f"RAG error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving relevant content"
        )
    except AIClientError as e:
        logger.error(f"AI client error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service temporarily unavailable"
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while processing query"
        )