"""
APIKey model for the AI-Native Book RAG Chatbot application.
"""
from sqlalchemy import Column, String, Boolean, Integer, DateTime
from sqlalchemy.orm import relationship
from src.models.base import Base


class APIKey(Base):
    """
    API key management for authentication (future implementation).
    """
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, index=True)  # UUID as string
    key_hash = Column(String(255), nullable=False, unique=True)  # Hashed API key
    name = Column(String(100))  # Key name/description
    user_id = Column(String(36))  # User who owns this key
    is_active = Column(Boolean, default=True)  # Whether key is currently valid
    rate_limit_requests = Column(Integer, default=1000)  # Requests per minute limit
    expires_at = Column(DateTime)  # Key expiration timestamp (optional)

    # Relationships would be defined if we had a User model