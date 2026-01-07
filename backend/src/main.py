"""
Main application file for the AI-Native Book RAG Chatbot application.
Sets up FastAPI app with all routes and middleware.
"""
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.utils.logging_config import setup_logging
from src.api import rag_api
from src.database.database import engine, Base
from src.utils.logging_config import get_logger
from src.utils.middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from src.utils.exceptions import BaseAppException

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

    # Validate configuration
    try:
        from src.config import validate_settings
        validate_settings()
        logger.info("Configuration validation passed")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        logger.error("Please check your environment variables and configuration")
        raise

    # Create database tables
    try:
        # Test database connection with a simple query
        from sqlalchemy import text
        with engine.connect() as conn:
            # Execute a simple query to test the connection
            result = conn.execute(text("SELECT 1"))
            logger.info("Database connection established successfully")

        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Provide more specific guidance for database connection issues
        logger.error("DATABASE CONNECTION FAILED - Please check your DATABASE_URL configuration")
        logger.error("For Neon Serverless PostgreSQL, ensure your DATABASE_URL is properly set")
        raise

    yield

    logger.info("Shutting down AI-Native Book RAG Chatbot application")


# Create FastAPI app with lifespan
app = FastAPI(
    title="AI-Native Book RAG Chatbot API",
    description="API for the AI-Native Book RAG Chatbot on Physical AI & Humanoid Robotics",
    version="1.0.0",
    lifespan=lifespan,
    redirect_slashes=True,  # Keep redirects but handle them properly
    redoc_url="/redoc",
    docs_url="/docs"
)

# Add middleware in order (last added = first executed)
# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# CORS middleware
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


# Exception handler for custom application exceptions
@app.exception_handler(BaseAppException)
async def app_exception_handler(request: Request, exc: BaseAppException):
    """
    Handler for custom application exceptions.

    Args:
        request: Request object
        exc: BaseAppException instance

    Returns:
        Error response with appropriate status code
    """
    logger.error(f"Application exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code or 500,
        content={
            "detail": exc.message,
            "error_type": exc.__class__.__name__,
            "status_code": exc.status_code or 500
        }
    )


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.

    Args:
        request: Request object
        exc: Exception object

    Returns:
        Error response
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "status_code": 500,
            "error_type": "InternalServerError"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Use 0.0.0.0 to bind to all interfaces for deployment
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )