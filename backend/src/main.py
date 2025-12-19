"""
Main application file for the AI-Native Book RAG Chatbot application.
Sets up FastAPI app with all routes and middleware.
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.utils.logging_config import setup_logging
from src.api import rag_api
from src.database.database import engine, Base
from src.utils.logging_config import get_logger

# Import all models to ensure they're registered with SQLAlchemy Base
from src.models.book_content import BookContent
from src.models.conversation import Conversation
from src.models.user_query import UserQuery
from src.models.chatbot_response import ChatbotResponse
from src.models.embedding import Embedding
from src.models.module import Module
from src.models.api_key import APIKey

# Setup logging based on settings
setup_logging(debug=settings.debug)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    logger.info("Starting up AI-Native Book RAG Chatbot application")

    # Small delay to ensure all models are loaded
    import time
    time.sleep(1)

    # Create database tables
    try:
        # Ensure the database connection works
        with engine.connect() as conn:
            logger.info("Database connection established successfully")

        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

    yield

    logger.info("Shutting down AI-Native Book RAG Chatbot application")


# Create FastAPI app with lifespan
app = FastAPI(
    title="AI-Native Book RAG Chatbot API",
    description="API for the AI-Native Book RAG Chatbot on Physical AI & Humanoid Robotics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose custom headers if needed
    # expose_headers=["Access-Control-Allow-Origin"]
)

# Include API routes
app.include_router(rag_api)


@app.get("/")
async def root():
    """
    Root endpoint for health check.

    Returns:
        Health check response
    """
    return {
        "message": "AI-Native Book RAG Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status response
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "service": "AI-Native Book RAG Chatbot API"
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for the application.

    Args:
        request: Request object
        exc: Exception object

    Returns:
        Error response
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {"detail": "Internal server error", "status_code": 500}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Use 0.0.0.0 to bind to all interfaces for deployment
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )