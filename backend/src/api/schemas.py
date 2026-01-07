"""
Pydantic schemas for request and response validation.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


# Request Models
class CreateConversationRequest(BaseModel):
    """Request model for creating a conversation."""
    user_id: Optional[str] = Field(None, max_length=255, description="Optional user identifier")
    title: Optional[str] = Field(None, max_length=500, description="Optional conversation title")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v):
        """Validate and sanitize title."""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                return None
        return v


class ProcessQueryRequest(BaseModel):
    """Request model for processing a query."""
    query_text: str = Field(..., min_length=1, max_length=5000, description="User's query text")
    selected_text: Optional[str] = Field(None, max_length=10000, description="Optional selected text for context")
    context_window: int = Field(5, ge=1, le=20, description="Number of surrounding chunks to include")

    @field_validator("query_text")
    @classmethod
    def validate_query_text(cls, v):
        """Validate and sanitize query text."""
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Query text cannot be empty")
        return v

    @field_validator("selected_text")
    @classmethod
    def validate_selected_text(cls, v):
        """Validate selected text."""
        if v is not None:
            v = v.strip()
            if len(v) == 0:
                return None
        return v


class EmbedContentRequest(BaseModel):
    """Request model for embedding content."""
    content: str = Field(..., min_length=1, max_length=50000, description="Content to embed")

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        """Validate content."""
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Content cannot be empty")
        return v


class QueryRequest(BaseModel):
    """Request model for legacy query endpoint."""
    query: str = Field(..., min_length=1, max_length=5000, description="User's query")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate query."""
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Query cannot be empty")
        return v


class QuerySelectedRequest(BaseModel):
    """Request model for query with selected text."""
    query: str = Field(..., min_length=1, max_length=5000, description="User's query")
    selected_text: str = Field(..., min_length=1, max_length=10000, description="Selected text for context")

    @field_validator("query", "selected_text")
    @classmethod
    def validate_fields(cls, v):
        """Validate fields."""
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Field cannot be empty")
        return v


class UpdateTitleRequest(BaseModel):
    """Request model for updating conversation title."""
    title: str = Field(..., min_length=1, max_length=500, description="New conversation title")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v):
        """Validate title."""
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Title cannot be empty")
        return v


# Response Models
class ConversationResponse(BaseModel):
    """Response model for conversation."""
    id: str
    session_id: str
    user_id: Optional[str]
    title: str
    is_active: Optional[bool] = True
    created_at: datetime
    updated_at: datetime
    conversation_metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """Response model for conversation list."""
    conversations: List[ConversationResponse]
    total: int
    skip: int = 0
    limit: int = 20


class QueryResponse(BaseModel):
    """Response model for query response."""
    response: str = Field(..., description="AI generated response")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations")


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""
    conversation_id: str
    history: List[Dict[str, Any]]
    total_messages: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")
    version: Optional[str] = "1.0.0"
    dependencies: Optional[Dict[str, str]] = Field(None, description="Dependency statuses")


class SuccessResponse(BaseModel):
    """Generic success response model."""
    success: bool = True
    message: str = Field(..., description="Success message")


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error detail")
    status_code: Optional[int] = None
    error_type: Optional[str] = None

