"""
Module model for the AI-Native Book RAG Chatbot application.
"""
from sqlalchemy import Column, String, Text, Integer
from sqlalchemy.orm import relationship
from .base import Base


class Module(Base):
    """
    Represents one of the 4 main curriculum modules.
    """
    __tablename__ = "modules"

    id = Column(String(50), primary_key=True, index=True)  # e.g., "module-1-ros2"
    title = Column(String(200), nullable=False)
    description = Column(Text)
    order_index = Column(Integer, nullable=False)  # Order in curriculum (1-4)
    estimated_duration_hours = Column(Integer)  # Estimated time to complete
    prerequisites = Column(String)  # JSON string for prerequisites
    learning_objectives = Column(String)  # JSON string for learning objectives

    # Relationships
    book_contents = relationship("BookContent", back_populates="module")


# Validation for order_index would be implemented in the service layer
MIN_MODULE_ORDER = 1
MAX_MODULE_ORDER = 4