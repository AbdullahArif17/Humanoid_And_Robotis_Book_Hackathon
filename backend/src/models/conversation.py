"""
Conversation model for the AI-Native Book RAG Chatbot application.
"""
from sqlalchemy import Column, String, Text, Boolean, JSON
from sqlalchemy.orm import relationship
from .base import Base


class Conversation(Base):
    """
    Represents a series of interactions between user and chatbot.
    """
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True, index=True)  # UUID as string
    session_id = Column(String(36), nullable=False, unique=True, index=True)  # Session identifier
    user_id = Column(String(36), nullable=True, index=True)  # User identifier
    title = Column(String(200))  # Conversation title (auto-generated from first query)
    is_active = Column(Boolean, default=True)  # Whether conversation is currently active
    metadata = Column(JSON)  # Additional conversation metadata

    # Relationships
    # Note: The relationship to UserQuery is through the session_id in UserQuery