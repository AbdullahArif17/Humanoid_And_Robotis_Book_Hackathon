#!/usr/bin/env python3
"""
Database Migration Script for AI-Native Book RAG Chatbot
This script adds missing timestamp columns to existing tables.
"""
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import text
from src.config import get_settings
from src.database.database import engine
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def run_migration():
    """Run the database migration to add missing timestamp columns."""
    print("Running database migration to add missing timestamp columns...")

    try:
        # Get database settings to show which database we're connecting to
        settings = get_settings()
        print(f"Connecting to database: {settings.database_url.replace('@', '***@') if '@' in settings.database_url else settings.database_url}")

        # Read the migration SQL file
        migration_file = Path(__file__).parent / "add_missing_timestamps.sql"
        if not migration_file.exists():
            print(f"[ERROR] Migration file not found: {migration_file}")
            return False

        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_sql = f.read()

        # Execute the migration
        with engine.connect() as conn:
            # Execute the migration SQL
            conn.execute(text(migration_sql))
            conn.commit()  # Commit the transaction

        print("[SUCCESS] Database migration completed successfully!")
        return True

    except Exception as e:
        print(f"[ERROR] Error running database migration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_migration()
    if success:
        print("\nDatabase migration completed successfully!")
        sys.exit(0)
    else:
        print("\nDatabase migration failed!")
        sys.exit(1)