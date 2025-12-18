"""
Hugging Face Spaces entry point for the AI-Native Book RAG Chatbot application.
This file serves as the entry point for Hugging Face Spaces deployment.
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

# Check if src directory exists
src_path = os.path.join(os.path.dirname(__file__), 'src')
if not os.path.exists(src_path):
    raise FileNotFoundError(f"src directory not found at {src_path}")

main_path = os.path.join(src_path, 'main.py')
if not os.path.exists(main_path):
    raise FileNotFoundError(f"main.py not found at {main_path}")

# Now try to import using importlib to bypass module system issues
import importlib.util

# Load main.py as a module
spec = importlib.util.spec_from_file_location("main_module", main_path)
main_module = importlib.util.module_from_spec(spec)

# Add it to sys.modules to make it importable
sys.modules["main_module"] = main_module

# Execute the module to load all definitions
spec.loader.exec_module(main_module)

# Now get the app instance
app = main_module.app