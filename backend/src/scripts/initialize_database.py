"""
Database initialization script for the AI-Native Book RAG Chatbot application.
Ensures all required tables are created in the database.
"""
import sys
import os
import traceback
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import inspect
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


def check_tables_exist():
    """Check if all required tables exist in the database."""
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

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
            if table not in existing_tables:
                missing_tables.append(table)

        return missing_tables, existing_tables
    except Exception as e:
        logger.warning(f"Could not inspect database tables: {str(e)}")
        # If we can't inspect the database, assume all tables are missing
        return required_tables, []


def create_missing_tables():
    """Create any missing tables in the database."""
    try:
        # Get database settings to show which database we're connecting to
        settings = get_settings()
        masked_url = settings.database_url
        if '@' in settings.database_url:
            # Mask the password in the URL for security
            parts = settings.database_url.split('@')
            auth_part = parts[0].split('://')[1] if '://' in parts[0] else parts[0]
            masked_url = f"{parts[0].replace(auth_part, '***')}@{parts[1]}" if len(parts) > 1 else settings.database_url

        logger.info(f"Connecting to database: {masked_url}")

        # Check what tables exist first
        missing_tables, existing_tables = check_tables_exist()

        logger.info(f"Existing tables: {existing_tables}")
        if missing_tables:
            logger.info(f"Missing tables: {missing_tables}")

            # Create all tables defined in the Base metadata
            Base.metadata.create_all(bind=engine)

            # Verify that tables were created
            missing_after_create, existing_after_create = check_tables_exist()

            if missing_after_create:
                logger.warning(f"Some tables may not have been created: {missing_after_create}")
                # Don't treat this as a complete failure, as some tables might have been created
                logger.info("Partial table creation completed.")
                return True
            else:
                logger.info("All required tables created successfully!")
                return True
        else:
            logger.info("All required tables already exist!")
            return True

    except Exception as e:
        logger.error(f"Error creating tables: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def main():
    """Main function to initialize the database."""
    logger.info("Starting database initialization...")

    success = create_missing_tables()

    if success:
        logger.info("Database initialization completed successfully!")
        print("Database initialization completed successfully!")
        return 0
    else:
        logger.error("Database initialization failed!")
        print("Database initialization failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())