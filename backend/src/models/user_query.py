"""
UserQuery model for the AI-Native Book RAG Chatbot application.
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base


class UserQuery(Base):
    """
    Represents questions from users to the AI system.
    """
    __tablename__ = "user_queries"

    id = Column(String(36), primary_key=True, index=True)  # UUID as string
    session_id = Column(String(36), nullable=False, index=True)  # Session identifier for conversation context
    user_id = Column(String(36), nullable=True, index=True)  # User identifier (null for anonymous users)
    query_text = Column(Text, nullable=False)
    query_type = Column(String(20), nullable=False)  # Enum: full_book, selected_text
    selected_text = Column(Text)  # Text selected by user for targeted queries (optional)
    query_context = Column(JSON)  # Additional context for the query
    processed_at = Column(DateTime)  # Processing completion timestamp (optional)

    # Relationships
    chatbot_responses = relationship("ChatbotResponse", back_populates="user_query")


# Validation for query_type would be implemented in the service layer
ALLOWED_QUERY_TYPES = ["full_book", "selected_text"]