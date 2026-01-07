"""
Input validation utilities for the AI-Native Book RAG Chatbot application.
"""
import re
from typing import List, Optional
from uuid import UUID


def contains_sql_injection_pattern(text: str) -> bool:
    """
    Check if text contains SQL injection patterns.
    
    Args:
        text: Text to check
        
    Returns:
        True if potentially harmful patterns found
    """
    if not isinstance(text, str):
        return False
    
    # Simple pattern matching for common SQL injection attempts
    sql_patterns = [
        r"(?i)(union\s+select)",
        r"(?i)(drop\s+table)",
        r"(?i)(delete\s+from)",
        r"(?i)(insert\s+into)",
        r"(?i)(update\s+.*\s+set)",
        r"(?i)(exec\s*\()",
        r"(?i)(--\s)",
        r"(?i)(/\*.*\*/)",
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def validate_uuid(uuid_string: str) -> bool:
    """
    Validate if a string is a valid UUID.
    
    Args:
        uuid_string: String to validate
        
    Returns:
        True if valid UUID, False otherwise
    """
    try:
        UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False


def validate_query_text(query: str, min_length: int = 1, max_length: int = 5000) -> List[str]:
    """
    Validate query text.
    
    Args:
        query: Query text to validate
        min_length: Minimum length
        max_length: Maximum length
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(query, str):
        errors.append("Query must be a string")
        return errors
    
    query = query.strip()
    
    if len(query) < min_length:
        errors.append(f"Query must be at least {min_length} characters long")
    
    if len(query) > max_length:
        errors.append(f"Query must be at most {max_length} characters long")
    
    # Check for potentially harmful content
    if contains_sql_injection_pattern(query):
        errors.append("Query contains potentially harmful content")
    
    return errors


def validate_selected_text(selected_text: str, max_length: int = 10000) -> List[str]:
    """
    Validate selected text.
    
    Args:
        selected_text: Selected text to validate
        max_length: Maximum length
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(selected_text, str):
        errors.append("Selected text must be a string")
        return errors
    
    selected_text = selected_text.strip()
    
    if len(selected_text) == 0:
        errors.append("Selected text cannot be empty")
    
    if len(selected_text) > max_length:
        errors.append(f"Selected text must be at most {max_length} characters long")
    
    return errors


def validate_conversation_id(conversation_id: str) -> List[str]:
    """
    Validate conversation ID.
    
    Args:
        conversation_id: Conversation ID to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(conversation_id, str):
        errors.append("Conversation ID must be a string")
        return errors
    
    if len(conversation_id.strip()) == 0:
        errors.append("Conversation ID cannot be empty")
    
    if not validate_uuid(conversation_id):
        errors.append("Conversation ID must be a valid UUID")
    
    return errors


def validate_user_id(user_id: Optional[str]) -> List[str]:
    """
    Validate user ID.
    
    Args:
        user_id: User ID to validate (optional)
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if user_id is not None:
        if not isinstance(user_id, str):
            errors.append("User ID must be a string")
        elif len(user_id.strip()) == 0:
            errors.append("User ID cannot be empty if provided")
        elif len(user_id) > 255:
            errors.append("User ID must be at most 255 characters long")
    
    return errors


def validate_title(title: Optional[str], max_length: int = 500) -> List[str]:
    """
    Validate conversation title.
    
    Args:
        title: Title to validate (optional)
        max_length: Maximum length
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if title is not None:
        if not isinstance(title, str):
            errors.append("Title must be a string")
        elif len(title.strip()) > max_length:
            errors.append(f"Title must be at most {max_length} characters long")
    
    return errors


def validate_pagination_params(skip: int, limit: int, max_limit: int = 100) -> List[str]:
    """
    Validate pagination parameters.
    
    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        max_limit: Maximum allowed limit
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not isinstance(skip, int) or skip < 0:
        errors.append("Skip must be a non-negative integer")
    
    if not isinstance(limit, int) or limit < 1:
        errors.append("Limit must be a positive integer")
    elif limit > max_limit:
        errors.append(f"Limit must be at most {max_limit}")
    
    return errors


def sanitize_input(text: str) -> str:
    """
    Sanitize user input by removing potentially harmful characters.
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove null bytes
    text = text.replace("\x00", "")
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text.strip()

