"""
Conversation model for the AI-Native Book RAG Chatbot application.
"""
from datetime import datetime
from sqlalchemy import Column, String, Text, Boolean, JSON, DateTime
from sqlalchemy.orm import relationship
from src.models.base import Base


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
    created_at = Column(DateTime, default=datetime.utcnow)  # Creation timestamp
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # Update timestamp
    conversation_metadata = Column(JSON)  # Additional conversation metadata

    # Relationships
    user_queries = relationship("UserQuery", back_populates="conversation")