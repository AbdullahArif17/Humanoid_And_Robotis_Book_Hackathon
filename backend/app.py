"""
Hugging Face Spaces entry point for the AI-Native Book RAG Chatbot application.
This file serves as the entry point for Hugging Face Spaces deployment.
"""
import sys
import os

# Add the current directory to Python path to ensure 'src' can be imported
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Add the parent directory to Python path as well
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import the app directly from src.main
try:
    from src.main import app
except ImportError as e:
    import traceback
    traceback.print_exc()
    raise