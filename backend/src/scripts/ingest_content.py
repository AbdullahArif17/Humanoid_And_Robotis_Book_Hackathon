"""
Content ingestion pipeline for the AI-Native Book RAG Chatbot application.
Parses book content from Docusaurus markdown files and populates the database and vector store.
"""
import os
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import asyncio
from sqlalchemy.orm import Session

from ..database.database import get_db, engine
from ..models.book_content import BookContent, ALLOWED_CONTENT_TYPES
from ..models.module import Module
from ..services.book_content_service import BookContentService
from ..vector_store.qdrant_client import QdrantClientWrapper
from ..ai.openai_client import get_openai_client
from ..utils.logging_config import get_logger
from ..utils.exceptions import ValidationError

logger = get_logger(__name__)


class ContentIngestor:
    """
    Content ingestion pipeline that processes Docusaurus markdown files
    and populates the database and vector store.
    """

    def __init__(self, db_session: Session):
        """
        Initialize the content ingestor.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self.book_content_service = BookContentService(db_session=db_session)
        self.qdrant_client = QdrantClientWrapper()
        self.openai_client = get_openai_client()

    def extract_frontmatter(self, content: str) -> tuple:
        """
        Extract frontmatter from markdown content.

        Args:
            content: Raw markdown content

        Returns:
            Tuple of (frontmatter_dict, content_without_frontmatter)
        """
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = parts[1]
                content_body = parts[2]

                # Parse frontmatter (simplified - in a real implementation, use yaml library)
                frontmatter_dict = {}
                for line in frontmatter.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().strip('"\'')
                        frontmatter_dict[key] = value

                return frontmatter_dict, content_body

        return {}, content

    def parse_markdown_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Parse a markdown file and extract content information.

        Args:
            file_path: Path to the markdown file

        Returns:
            Dictionary with content information
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        frontmatter, content_body = self.extract_frontmatter(content)

        # Extract module and section info from path
        # Expected path format: book/docs/module-X-name/section-name.md
        path_parts = file_path.parts
        module_dir_index = -1
        for i, part in enumerate(path_parts):
            if part.startswith('module-'):
                module_dir_index = i
                break

        if module_dir_index == -1:
            raise ValidationError(f"File {file_path} is not in a module directory")

        module_id = path_parts[module_dir_index]
        section_path = '/'.join(path_parts[module_dir_index:]).replace('.md', '').replace('.markdown', '')

        # Extract title from frontmatter or first heading
        title = frontmatter.get('sidebar_label', frontmatter.get('title', ''))
        if not title:
            # Look for first heading in content
            first_heading_match = re.search(r'^#\s+(.+)$', content_body, re.MULTILINE)
            if first_heading_match:
                title = first_heading_match.group(1).strip()
            else:
                title = file_path.stem.replace('-', ' ').title()

        # Determine content type based on file characteristics
        content_type = 'text'  # Default
        if '```' in content_body:
            content_type = 'code'
        elif any(keyword in content_body.lower() for keyword in ['diagram', 'graph', 'chart']):
            content_type = 'diagram'

        # Create metadata
        metadata = {
            'source_file': str(file_path),
            'file_last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'word_count': len(content_body.split()),
            'has_code_blocks': '```' in content_body,
            'has_images': '![alt text]' in content_body or '<img' in content_body
        }

        # Add any frontmatter properties to metadata
        for key, value in frontmatter.items():
            if key not in ['title', 'sidebar_label', 'sidebar_position']:
                metadata[key] = value

        return {
            'title': title,
            'module_id': module_id,
            'section_path': section_path,
            'content_type': content_type,
            'content_body': content_body,
            'metadata': metadata
        }

    def process_directory(self, directory_path: Path) -> List[Dict[str, Any]]:
        """
        Process all markdown files in a directory recursively.

        Args:
            directory_path: Path to the directory to process

        Returns:
            List of content dictionaries
        """
        contents = []

        for file_path in directory_path.rglob('*'):
            if file_path.suffix.lower() in ['.md', '.markdown']:
                try:
                    content_info = self.parse_markdown_file(file_path)
                    contents.append(content_info)
                    logger.info(f"Parsed content from {file_path}")
                except Exception as e:
                    logger.error(f"Error parsing {file_path}: {str(e)}")

        return contents

    def ingest_content(self, content_info: Dict[str, Any]) -> bool:
        """
        Ingest a single piece of content into the database and vector store.

        Args:
            content_info: Dictionary with content information

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if content already exists
            existing_content = self.book_content_service.get_content_by_section_path(
                content_info['section_path']
            )

            if existing_content:
                logger.info(f"Content already exists: {content_info['section_path']}")
                # Optionally update existing content
                # self.book_content_service.update_content(
                #     existing_content.id,
                #     title=content_info['title'],
                #     content_body=content_info['content_body'],
                #     metadata=content_info['metadata']
                # )
                return True

            # Create content in database
            content = self.book_content_service.create_content(
                title=content_info['title'],
                module_id=content_info['module_id'],
                section_path=content_info['section_path'],
                content_type=content_info['content_type'],
                content_body=content_info['content_body'],
                metadata=content_info['metadata']
            )

            # Create embedding and store in vector database
            embedding = self.openai_client.generate_embeddings([content_info['content_body']])[0]

            self.qdrant_client.store_embedding(
                content_id=content.id,
                title=content_info['title'],
                section_path=content_info['section_path'],
                content_body=content_info['content_body'],
                embedding=embedding,
                metadata=content_info['metadata']
            )

            logger.info(f"Successfully ingested content: {content_info['section_path']}")
            return True

        except Exception as e:
            logger.error(f"Error ingesting content {content_info['section_path']}: {str(e)}")
            return False

    def run_ingestion(self, book_path: str) -> int:
        """
        Run the full content ingestion pipeline.

        Args:
            book_path: Path to the book directory

        Returns:
            Number of successfully ingested content items
        """
        book_dir = Path(book_path)
        if not book_dir.exists():
            raise ValueError(f"Book directory does not exist: {book_path}")

        logger.info(f"Starting content ingestion from {book_path}")

        # Process all markdown files
        contents = self.process_directory(book_dir / 'docs')

        success_count = 0
        for content_info in contents:
            if self.ingest_content(content_info):
                success_count += 1

        logger.info(f"Content ingestion completed. Successfully ingested {success_count}/{len(contents)} items")
        return success_count


def main():
    """
    Main function to run the content ingestion pipeline.
    """
    # Set up logging
    from ..utils.logging_config import setup_logging
    setup_logging(debug=True)

    # Get database session
    db = next(get_db())

    try:
        # Create ingestor and run
        ingestor = ContentIngestor(db_session=db)

        # Get book path from command line or use default
        book_path = sys.argv[1] if len(sys.argv) > 1 else "../../../book"

        success_count = ingestor.run_ingestion(book_path)

        print(f"Ingestion complete. Successfully processed {success_count} content items.")

    except Exception as e:
        logger.error(f"Error running content ingestion: {str(e)}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()