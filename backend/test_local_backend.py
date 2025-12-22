#!/usr/bin/env python3
"""
Local test to check if the backend code is properly configured and can start without errors.
This script tests the backend code structure without requiring deployment.
"""
import sys
import os
import importlib.util
from pathlib import Path

def test_imports():
    """
    Test if all backend modules can be imported without errors.

    Returns:
        bool: True if all imports succeed, False otherwise
    """
    print("[INFO] Testing Module Imports")
    print("-" * 40)

    backend_path = Path(__file__).parent
    src_path = backend_path / "src"

    modules_to_test = [
        "src.config",
        "src.main",
        "src.api.rag_api",
        "src.services.chat_service",
        "src.models.conversation",
        "src.models.user_query",
        "src.models.chatbot_response",
        "src.database.database",
        "src.ai.openai_client",
        "src.vector_store.qdrant_client",
    ]

    all_passed = True

    for module_name in modules_to_test:
        try:
            print(f"Testing import: {module_name}")
            # Add src to Python path to allow imports
            sys.path.insert(0, str(src_path))
            importlib.import_module(module_name)
            print(f"[SUCCESS] {module_name} - Import successful")
        except ImportError as e:
            print(f"[ERROR] {module_name} - Import failed: {e}")
            all_passed = False
        except Exception as e:
            print(f"[ERROR] {module_name} - Error: {e}")
            all_passed = False

    return all_passed

def test_config():
    """
    Test if the configuration can be loaded without errors.

    Returns:
        bool: True if config loads successfully, False otherwise
    """
    print("\n[INFO] Testing Configuration")
    print("-" * 40)

    try:
        from src.config import settings, get_settings
        print("[SUCCESS] Configuration module loaded successfully")

        # Test that we can access settings
        print(f"[SUCCESS] OpenAI model setting: {settings.openai_model}")
        print(f"[SUCCESS] Database URL setting: {settings.database_url}")
        print(f"[SUCCESS] Port setting: {settings.port}")
        print("[SUCCESS] All settings accessible")

        # Test get_settings function
        settings_func = get_settings()
        print("[SUCCESS] get_settings() function works")

        return True
    except Exception as e:
        print(f"[ERROR] Configuration test failed: {e}")
        return False

def test_models():
    """
    Test if database models are properly defined.

    Returns:
        bool: True if models are defined correctly, False otherwise
    """
    print("\n[INFO] Testing Database Models")
    print("-" * 40)

    try:
        from src.models.conversation import Conversation
        from src.models.user_query import UserQuery
        from src.models.chatbot_response import ChatbotResponse
        from src.models.base import Base

        print("[SUCCESS] Models imported successfully")

        # Check if models have expected attributes
        conversation_attrs = ['id', 'session_id', 'user_id', 'title', 'created_at', 'updated_at', 'conversation_metadata']
        for attr in conversation_attrs:
            if hasattr(Conversation, attr):
                print(f"[SUCCESS] Conversation model has '{attr}' attribute")
            else:
                print(f"[ERROR] Conversation model missing '{attr}' attribute")
                return False

        user_query_attrs = ['id', 'session_id', 'query_text', 'query_type', 'conversation_id']
        for attr in user_query_attrs:
            if hasattr(UserQuery, attr):
                print(f"[SUCCESS] UserQuery model has '{attr}' attribute")
            else:
                print(f"[ERROR] UserQuery model missing '{attr}' attribute")
                return False

        response_attrs = ['id', 'query_id', 'response_text']
        for attr in response_attrs:
            if hasattr(ChatbotResponse, attr):
                print(f"[SUCCESS] ChatbotResponse model has '{attr}' attribute")
            else:
                print(f"[ERROR] ChatbotResponse model missing '{attr}' attribute")
                return False

        print("[SUCCESS] All models have expected attributes")
        return True

    except Exception as e:
        print(f"[ERROR] Model test failed: {e}")
        return False

def test_services():
    """
    Test if service classes are properly defined.

    Returns:
        bool: True if services are defined correctly, False otherwise
    """
    print("\n[INFO] Testing Service Classes")
    print("-" * 40)

    try:
        from src.services.chat_service import ChatService
        print("[SUCCESS] ChatService class loaded successfully")

        # Check if the class has expected methods
        expected_methods = [
            'create_conversation',
            'get_conversation',
            'process_query'
        ]

        for method in expected_methods:
            if hasattr(ChatService, method):
                print(f"[SUCCESS] ChatService has '{method}' method")
            else:
                print(f"[ERROR] ChatService missing '{method}' method")
                return False

        print("[SUCCESS] All service methods present")
        return True

    except Exception as e:
        print(f"[ERROR] Service test failed: {e}")
        return False

def test_app_creation():
    """
    Test if the FastAPI app can be created without errors.

    Returns:
        bool: True if app creation succeeds, False otherwise
    """
    print("\n[INFO] Testing FastAPI App Creation")
    print("-" * 40)

    try:
        # Temporarily add src to path
        src_path = Path(__file__).parent / "src"
        sys.path.insert(0, str(src_path))

        from src.main import app
        print("[SUCCESS] FastAPI app created successfully")

        # Check if app has expected routes
        route_paths = [route.path for route in app.routes]
        expected_paths = ["/", "/health", "/api/query", "/api/query-selected", "/api/embed"]

        for path in expected_paths:
            if any(path in route_path for route_path in route_paths):
                print(f"[SUCCESS] Route '{path}' is registered")
            else:
                print(f"[WARNING] Route '{path}' may not be registered (not necessarily an error)")

        print("[SUCCESS] App creation test completed")
        return True

    except Exception as e:
        print(f"[ERROR] App creation test failed: {e}")
        return False

def main():
    """
    Main function to run all local backend tests.
    """
    print("[INFO] LOCAL BACKEND CODE VERIFICATION TEST")
    print("=" * 60)

    print("This test verifies that your backend code is properly structured")
    print("and can be imported without errors. It does not require deployment.\n")

    # Run all tests
    imports_ok = test_imports()
    config_ok = test_config()
    models_ok = test_models()
    services_ok = test_services()
    app_ok = test_app_creation()

    print("\n" + "=" * 60)
    print("[INFO] LOCAL TEST RESULTS:")
    print(f"Module imports: {'[SUCCESS] PASSED' if imports_ok else '[ERROR] FAILED'}")
    print(f"Configuration: {'[SUCCESS] PASSED' if config_ok else '[ERROR] FAILED'}")
    print(f"Database models: {'[SUCCESS] PASSED' if models_ok else '[ERROR] FAILED'}")
    print(f"Service classes: {'[SUCCESS] PASSED' if services_ok else '[ERROR] FAILED'}")
    print(f"App creation: {'[SUCCESS] PASSED' if app_ok else '[ERROR] FAILED'}")

    all_tests_passed = all([imports_ok, config_ok, models_ok, services_ok, app_ok])

    if all_tests_passed:
        print(f"\n[SUCCESS] All local tests passed! Your backend code is properly structured.")
        print("This suggests that if environment variables are correctly set,")
        print("the backend should start and respond correctly when deployed.")
        return True
    else:
        print(f"\n[ERROR] Some local tests failed. Please fix the issues above")
        print("before deploying your backend.")
        return False

if __name__ == "__main__":
    main()