"""
Hugging Face Spaces entry point for the AI-Native Book RAG Chatbot application.
This file serves as the entry point for Hugging Face Spaces deployment.
"""
from src.main import app

# This is the entry point for Hugging Face Spaces
# Hugging Face Spaces will look for a variable called "app"
# The app is already created in src/main.py, so we just import and expose it

# The application is available at the "app" variable
# Hugging Face Spaces will automatically run this with their own uvicorn server