"""
Logging configuration for the AI-Native Book RAG Chatbot application.
"""
import logging
import sys
from datetime import datetime

# Try to import pythonjsonlogger, fallback to regular logging if not available
try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    # If pythonjsonlogger is not available, define a dummy class
    class jsonlogger:
        class JsonFormatter:
            def __init__(self, *args, **kwargs):
                # Fall back to regular formatter if jsonlogger not available
                self.fallback_formatter = logging.Formatter(*args)

            def format(self, record):
                return self.fallback_formatter.format(record)


def setup_logging(debug: bool = False):
    """
    Setup logging configuration for the application.

    Args:
        debug: Whether to enable debug logging
    """
    # Create formatters
    if debug:
        # More detailed format for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        )
    else:
        # JSON format for production
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s',
            rename_fields={'asctime': '@timestamp', 'levelname': 'level', 'name': 'logger'}
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)

    # Set specific log levels for external libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Name of the logger

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)