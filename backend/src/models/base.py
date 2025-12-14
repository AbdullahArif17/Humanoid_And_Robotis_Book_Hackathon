"""
Base model for all SQLAlchemy models in the AI-Native Book RAG Chatbot application.
"""
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, func
from sqlalchemy.orm import as_declarative, declared_attr
from datetime import datetime


@as_declarative()
class Base:
    """
    Base class for all SQLAlchemy models.
    Provides common fields like id, created_at, updated_at.
    """
    id: Column
    __name__: str

    @declared_attr
    def __tablename__(cls) -> str:
        return cls.__name__.lower()

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)