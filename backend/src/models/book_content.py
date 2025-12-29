"""
BookContent model for the AI-Native Book RAG Chatbot application.
"""
from sqlalchemy import Column, String, Text, Integer, Boolean, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .base import Base


class BookContent(Base):
    """
    Represents educational content organized into modules and sections.
    """
    __tablename__ = "book_content"

    id = Column(String(36), primary_key=True, index=True)  # UUID as string
    title = Column(String(255), nullable=False)
    module_id = Column(String(50), ForeignKey("modules.id"), nullable=False, index=True)  # e.g., "module-1-ros2"
    section_path = Column(String(200), nullable=False, index=True)  # e.g., "module-1-ros2/basics/nodes"
    content_type = Column(String(20), nullable=False)  # Enum: text, code, diagram, exercise, lab
    content_body = Column(Text, nullable=False)
    content_metadata = Column(JSON)  # Additional content metadata (tags, difficulty, prerequisites)
    version = Column(Integer, default=1)  # Content version for change tracking
    chunk_boundary = Column(Boolean, default=False)  # Whether this represents a semantic chunk boundary

    # Relationships
    embeddings = relationship("Embedding", back_populates="book_content")
    module = relationship("Module", back_populates="book_contents")


# Validation for content_type would be implemented in the service layer
ALLOWED_CONTENT_TYPES = ["text", "code", "diagram", "exercise", "lab"]