"""
Embedding model for the AI-Native Book RAG Chatbot application.
"""
from sqlalchemy import Column, String, Text, Integer, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base


class Embedding(Base):
    """
    Vector representation of book content for semantic search.
    """
    __tablename__ = "embeddings"

    id = Column(String(36), primary_key=True, index=True)  # UUID as string
    content_id = Column(String(36), ForeignKey("book_content.id"), nullable=False)  # Reference to BookContent
    # Note: We'll store the vector data in Qdrant, not in the database
    chunk_text = Column(Text, nullable=False)  # Original text that was embedded
    chunk_metadata = Column(JSON)  # Additional chunk metadata
    content_version = Column(Integer)  # Version of content at time of embedding

    # Relationships
    book_content = relationship("BookContent", back_populates="embeddings")