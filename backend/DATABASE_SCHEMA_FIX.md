# Database Schema Issue and Solution

## Problem
The database schema was missing `created_at` and `updated_at` columns in several tables, causing the following error when processing queries:

```
RAGError: Error processing query: (psycopg2.errors.UndefinedColumn) column "created_at" of relation "user_queries" does not exist
```

## Root Cause
The SQLAlchemy models inherit from a Base class that defines `created_at` and `updated_at` timestamp columns, but the SQL schema in `create_tables.sql` was missing these columns for several tables.

## Tables Affected
- `user_queries` - missing `created_at` and `updated_at`
- `chatbot_responses` - missing `created_at` and `updated_at`
- `book_content` - missing `created_at` and `updated_at`
- `modules` - missing `created_at` and `updated_at`
- `embeddings` - missing `created_at` and `updated_at`
- `api_keys` - missing `created_at` and `updated_at`

## Solution
1. **Updated Schema File**: Modified `create_tables.sql` to include the missing timestamp columns
2. **Migration Script**: Created `add_missing_timestamps.sql` to add columns to existing databases
3. **Migration Runner**: Created `run_migration.py` to execute the migration

## For New Installations
The updated `create_tables.sql` will create tables with the correct schema from the start.

## For Existing Databases
Run the migration script to add the missing columns:
```bash
python run_migration.py
```

## Columns Added
Each affected table now includes:
- `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
- `updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`

## Additional Improvements
The migration script also includes triggers to automatically update the `updated_at` column whenever a row is modified.