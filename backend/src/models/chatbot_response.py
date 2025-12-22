"""
ChatbotResponse model for the AI-Native Book RAG Chatbot application.
"""
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from src.models.base import Base


class ChatbotResponse(Base):
    """
    AI-generated responses to user queries with source attribution.
    """
    __tablename__ = "chatbot_responses"

    id = Column(String(36), primary_key=True, index=True)  # UUID as string
    query_id = Column(String(36), ForeignKey("user_queries.id"), nullable=False)  # Reference to original UserQuery
    response_text = Column(Text, nullable=False)
    sources = Column(JSON)  # References to book content that informed the response
    confidence_score = Column(Float)  # Confidence level of the response (0.0-1.0)
    tokens_used = Column(Integer)  # Number of tokens in the response
    model_used = Column(String(100))  # LLM model that generated the response
    timestamp = Column(DateTime, default=datetime.utcnow)  # Creation timestamp

    # Relationships
    user_query = relationship("UserQuery", back_populates="chatbot_responses")