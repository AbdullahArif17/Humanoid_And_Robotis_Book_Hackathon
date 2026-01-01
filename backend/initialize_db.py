#!/usr/bin/env python3
"""
Database Initialization Script for AI-Native Book RAG Chatbot
This script creates all required database tables in your Neon database.
"""
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import inspect, text
from src.config import get_settings
from src.database.database import engine, Base
from src.models.conversation import Conversation
from src.models.user_query import UserQuery
from src.models.chatbot_response import ChatbotResponse
from src.models.book_content import BookContent
from src.models.module import Module
from src.models.embedding import Embedding
from src.models.api_key import APIKey
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def initialize_database():
    """Initialize the database by creating all required tables."""
    print("Initializing database...")

    try:
        # Get database settings to show which database we're connecting to
        settings = get_settings()
        print(f"Connecting to database: {settings.database_url.replace('@', '***@') if '@' in settings.database_url else settings.database_url}")

        # Test database connection
        with engine.connect() as conn:
            # Execute a simple query to test the connection
            result = conn.execute(text("SELECT 1"))
            print("[SUCCESS] Database connection established successfully")

        # Check existing tables
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()
        print(f"Existing tables: {existing_tables}")

        # Create all tables
        print("Creating tables...")
        Base.metadata.create_all(bind=engine)
        print("[SUCCESS] Tables created successfully")

        # Verify tables were created
        new_existing_tables = inspector.get_table_names()
        print(f"Tables after creation: {new_existing_tables}")

        # Check for required tables
        required_tables = [
            'conversations',
            'user_queries',
            'chatbot_responses',
            'book_content',
            'modules',
            'embeddings',
            'api_keys'
        ]

        missing_tables = []
        for table in required_tables:
            if table not in new_existing_tables:
                missing_tables.append(table)

        if missing_tables:
            print(f"[ERROR] Missing tables: {missing_tables}")
            return False
        else:
            print("[SUCCESS] All required tables exist!")
            return True

    except Exception as e:
        print(f"[ERROR] Error initializing database: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = initialize_database()
    if success:
        print("\nDatabase initialization completed successfully!")
        sys.exit(0)
    else:
        print("\nDatabase initialization failed!")
        sys.exit(1)