"""
Custom exceptions for the AI-Native Book RAG Chatbot application.
"""
from typing import Optional


class BaseAppException(Exception):
    """Base exception class for the application."""

    def __init__(self, message: str, status_code: Optional[int] = None, detail: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.detail = detail

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"


class ValidationError(BaseAppException):
    """Raised when validation fails."""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(message, 400, detail)


class DatabaseError(BaseAppException):
    """Raised when database operations fail."""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(message, 500, detail)


class AIClientError(BaseAppException):
    """Raised when AI client operations fail."""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(message, 500, detail)


class RAGError(BaseAppException):
    """Raised when RAG operations fail."""

    def __init__(self, message: str, detail: Optional[str] = None):
        super().__init__(message, 500, detail)