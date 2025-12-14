"""
BookContentService for the AI-Native Book RAG Chatbot application.
Handles CRUD operations for book content with proper validation and error handling.
"""
import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import and_, or_

from ..models.book_content import BookContent, ALLOWED_CONTENT_TYPES
from ..models.module import Module
from ..utils.exceptions import ValidationError, DatabaseError
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class BookContentService:
    """
    Service class for handling BookContent operations.

    This service provides methods for:
    - Creating new content
    - Retrieving content by various criteria
    - Updating existing content
    - Deleting content
    - Validating content before operations
    """

    def __init__(self, db_session: Session):
        """
        Initialize the BookContentService with a database session.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session

    def create_content(
        self,
        title: str,
        module_id: str,
        section_path: str,
        content_type: str,
        content_body: str,
        metadata: Optional[Dict[str, Any]] = None,
        version: int = 1
    ) -> BookContent:
        """
        Create a new BookContent entry.

        Args:
            title: Content title
            module_id: Module identifier
            section_path: Path to the section (e.g., "module-1-ros2/basics/nodes")
            content_type: Type of content (text, code, diagram, exercise, lab)
            content_body: Main content body
            metadata: Additional metadata (tags, difficulty, prerequisites)
            version: Content version (default: 1)

        Returns:
            Created BookContent instance

        Raises:
            ValidationError: If input validation fails
            DatabaseError: If database operation fails
        """
        # Validate inputs
        self._validate_inputs(title, module_id, section_path, content_type, content_body)

        # Check if module exists
        module_exists = self.db_session.query(Module).filter(Module.id == module_id).first()
        if not module_exists:
            raise ValidationError(f"Module with ID '{module_id}' does not exist")

        # Check if content with same section_path already exists
        existing_content = self.db_session.query(BookContent).filter(
            BookContent.section_path == section_path
        ).first()
        if existing_content:
            raise ValidationError(f"Content with section path '{section_path}' already exists")

        # Create new content
        content = BookContent(
            id=str(uuid.uuid4()),
            title=title,
            module_id=module_id,
            section_path=section_path,
            content_type=content_type,
            content_body=content_body,
            metadata=metadata or {},
            version=version
        )

        try:
            self.db_session.add(content)
            self.db_session.commit()
            self.db_session.refresh(content)

            logger.info(f"Created new content: {content.id} - {content.title}")
            return content
        except IntegrityError as e:
            self.db_session.rollback()
            logger.error(f"Database integrity error creating content: {str(e)}")
            raise DatabaseError(f"Failed to create content: {str(e)}")
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Unexpected error creating content: {str(e)}")
            raise DatabaseError(f"Unexpected error creating content: {str(e)}")

    def get_content_by_id(self, content_id: str) -> Optional[BookContent]:
        """
        Retrieve content by its ID.

        Args:
            content_id: Unique identifier of the content

        Returns:
            BookContent instance or None if not found
        """
        try:
            content = self.db_session.query(BookContent).filter(BookContent.id == content_id).first()
            return content
        except Exception as e:
            logger.error(f"Error retrieving content by ID {content_id}: {str(e)}")
            raise DatabaseError(f"Error retrieving content: {str(e)}")

    def get_content_by_section_path(self, section_path: str) -> Optional[BookContent]:
        """
        Retrieve content by its section path.

        Args:
            section_path: Path to the section (e.g., "module-1-ros2/basics/nodes")

        Returns:
            BookContent instance or None if not found
        """
        try:
            content = self.db_session.query(BookContent).filter(
                BookContent.section_path == section_path
            ).first()
            return content
        except Exception as e:
            logger.error(f"Error retrieving content by section path {section_path}: {str(e)}")
            raise DatabaseError(f"Error retrieving content: {str(e)}")

    def get_content_by_module(self, module_id: str, content_type: Optional[str] = None) -> List[BookContent]:
        """
        Retrieve all content for a specific module, optionally filtered by content type.

        Args:
            module_id: Module identifier
            content_type: Optional content type filter

        Returns:
            List of BookContent instances
        """
        try:
            query = self.db_session.query(BookContent).filter(BookContent.module_id == module_id)

            if content_type:
                query = query.filter(BookContent.content_type == content_type)

            return query.all()
        except Exception as e:
            logger.error(f"Error retrieving content for module {module_id}: {str(e)}")
            raise DatabaseError(f"Error retrieving content: {str(e)}")

    def get_all_content(self, skip: int = 0, limit: int = 100) -> List[BookContent]:
        """
        Retrieve all content with pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of BookContent instances
        """
        try:
            return self.db_session.query(BookContent).offset(skip).limit(limit).all()
        except Exception as e:
            logger.error(f"Error retrieving all content: {str(e)}")
            raise DatabaseError(f"Error retrieving content: {str(e)}")

    def update_content(
        self,
        content_id: str,
        title: Optional[str] = None,
        module_id: Optional[str] = None,
        section_path: Optional[str] = None,
        content_type: Optional[str] = None,
        content_body: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        version: Optional[int] = None
    ) -> Optional[BookContent]:
        """
        Update an existing content entry.

        Args:
            content_id: Unique identifier of the content to update
            title: New title (if provided)
            module_id: New module ID (if provided)
            section_path: New section path (if provided)
            content_type: New content type (if provided)
            content_body: New content body (if provided)
            metadata: New metadata (if provided)
            version: New version (if provided)

        Returns:
            Updated BookContent instance or None if not found

        Raises:
            ValidationError: If input validation fails
            DatabaseError: If database operation fails
        """
        content = self.get_content_by_id(content_id)
        if not content:
            return None

        # Prepare updates
        updates = {}
        if title is not None:
            self._validate_title(title)
            updates['title'] = title
        if module_id is not None:
            # Check if module exists
            module_exists = self.db_session.query(Module).filter(Module.id == module_id).first()
            if not module_exists:
                raise ValidationError(f"Module with ID '{module_id}' does not exist")
            updates['module_id'] = module_id
        if section_path is not None:
            self._validate_section_path(section_path)
            # Check if another content with same section_path exists (excluding current content)
            existing_content = self.db_session.query(BookContent).filter(
                and_(
            BookContent.section_path == section_path,
            BookContent.id != content_id
        )
            ).first()
            if existing_content:
                raise ValidationError(f"Content with section path '{section_path}' already exists")
            updates['section_path'] = section_path
        if content_type is not None:
            self._validate_content_type(content_type)
            updates['content_type'] = content_type
        if content_body is not None:
            self._validate_content_body(content_body)
            updates['content_body'] = content_body
        if metadata is not None:
            updates['metadata'] = metadata
        if version is not None:
            updates['version'] = version

        # Update version if not explicitly provided
        if 'version' not in updates:
            updates['version'] = content.version + 1

        try:
            for key, value in updates.items():
                setattr(content, key, value)

            self.db_session.commit()
            self.db_session.refresh(content)

            logger.info(f"Updated content: {content.id} - {content.title}")
            return content
        except IntegrityError as e:
            self.db_session.rollback()
            logger.error(f"Database integrity error updating content: {str(e)}")
            raise DatabaseError(f"Failed to update content: {str(e)}")
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Unexpected error updating content: {str(e)}")
            raise DatabaseError(f"Unexpected error updating content: {str(e)}")

    def delete_content(self, content_id: str) -> bool:
        """
        Delete a content entry by its ID.

        Args:
            content_id: Unique identifier of the content to delete

        Returns:
            True if deletion was successful, False if content not found
        """
        content = self.get_content_by_id(content_id)
        if not content:
            return False

        try:
            self.db_session.delete(content)
            self.db_session.commit()

            logger.info(f"Deleted content: {content_id} - {content.title}")
            return True
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error deleting content {content_id}: {str(e)}")
            raise DatabaseError(f"Error deleting content: {str(e)}")

    def search_content(self, query: str, module_id: Optional[str] = None) -> List[BookContent]:
        """
        Search for content by title or content body.

        Args:
            query: Search query string
            module_id: Optional module ID to limit search

        Returns:
            List of matching BookContent instances
        """
        try:
            search_filter = or_(
                BookContent.title.contains(query),
                BookContent.content_body.contains(query)
            )

            query_obj = self.db_session.query(BookContent).filter(search_filter)

            if module_id:
                query_obj = query_obj.filter(BookContent.module_id == module_id)

            return query_obj.all()
        except Exception as e:
            logger.error(f"Error searching content: {str(e)}")
            raise DatabaseError(f"Error searching content: {str(e)}")

    def _validate_inputs(
        self,
        title: str,
        module_id: str,
        section_path: str,
        content_type: str,
        content_body: str
    ) -> None:
        """
        Validate all input parameters.

        Args:
            title: Content title
            module_id: Module identifier
            section_path: Section path
            content_type: Content type
            content_body: Content body

        Raises:
            ValidationError: If any validation fails
        """
        self._validate_title(title)
        self._validate_module_id(module_id)
        self._validate_section_path(section_path)
        self._validate_content_type(content_type)
        self._validate_content_body(content_body)

    def _validate_title(self, title: str) -> None:
        """Validate content title."""
        if not title or not title.strip():
            raise ValidationError("Title cannot be empty")
        if len(title.strip()) > 255:
            raise ValidationError("Title cannot exceed 255 characters")

    def _validate_module_id(self, module_id: str) -> None:
        """Validate module ID."""
        if not module_id or not module_id.strip():
            raise ValidationError("Module ID cannot be empty")

    def _validate_section_path(self, section_path: str) -> None:
        """Validate section path."""
        if not section_path or not section_path.strip():
            raise ValidationError("Section path cannot be empty")
        if len(section_path.strip()) > 200:
            raise ValidationError("Section path cannot exceed 200 characters")

    def _validate_content_type(self, content_type: str) -> None:
        """Validate content type."""
        if not content_type or not content_type.strip():
            raise ValidationError("Content type cannot be empty")
        if content_type not in ALLOWED_CONTENT_TYPES:
            raise ValidationError(f"Invalid content type. Allowed types: {ALLOWED_CONTENT_TYPES}")

    def _validate_content_body(self, content_body: str) -> None:
        """Validate content body."""
        if not content_body or not content_body.strip():
            raise ValidationError("Content body cannot be empty")