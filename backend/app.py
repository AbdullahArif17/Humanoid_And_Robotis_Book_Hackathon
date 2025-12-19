"""
Hugging Face Spaces entry point for the AI-Native Book RAG Chatbot application.
This file serves as the entry point for Hugging Face Spaces deployment.
"""
import sys
import os

# Add the current directory and src directory to Python path
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Add src to Python path for proper imports
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print(f"Files in current directory: {os.listdir(current_dir)}")
print(f"Files in src directory: {os.listdir(src_path)}")

# Check if src directory exists
if not os.path.exists(src_path):
    raise FileNotFoundError(f"src directory not found at {src_path}")

main_path = os.path.join(src_path, 'main.py')
if not os.path.exists(main_path):
    raise FileNotFoundError(f"main.py not found at {main_path}")

# Now try to import using importlib to bypass module system issues
import importlib.util

# Load main.py as a module
spec = importlib.util.spec_from_file_location("main_module", main_path)
if spec is None:
    raise ImportError(f"Could not load spec from {main_path}")

main_module = importlib.util.module_from_spec(spec)

# Add it to sys.modules to make it importable
sys.modules["main_module"] = main_module
sys.modules["src"] = main_module

# Execute the module to load all definitions
try:
    spec.loader.exec_module(main_module)
except Exception as e:
    print(f"Error executing main module: {e}")
    import traceback
    traceback.print_exc()
    raise

# Now get the app instance
if not hasattr(main_module, 'app'):
    raise AttributeError("main_module does not have 'app' attribute")

app = main_module.app