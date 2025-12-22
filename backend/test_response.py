#!/usr/bin/env python3
"""
Simple test to check if the backend is responding.
This script tests the basic connectivity to the backend API.
"""
import requests
import sys
import time

def test_backend_response(backend_url):
    """
    Test if the backend is responding to requests.

    Args:
        backend_url: The URL of the deployed backend

    Returns:
        bool: True if backend is responding, False otherwise
    """
    try:
        print(f"Testing backend response at: {backend_url}")

        # Test the root endpoint
        start_time = time.time()
        response = requests.get(backend_url, timeout=30)
        end_time = time.time()

        response_time = end_time - start_time

        print(f"Response status: {response.status_code}")
        print(f"Response time: {response_time:.2f} seconds")
        print(f"Response headers: {dict(response.headers)}")

        if response.status_code == 200:
            print("✅ Backend is responding successfully!")
            print(f"Response content: {response.text[:200]}...")
            return True
        else:
            print(f"❌ Backend returned status code: {response.status_code}")
            print(f"Response content: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Connection error - backend may not be running or accessible")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out - backend may be slow to respond")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed with error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_health_endpoint(backend_url):
    """
    Test the health endpoint specifically.

    Args:
        backend_url: The base URL of the deployed backend

    Returns:
        bool: True if health endpoint is responding, False otherwise
    """
    try:
        health_url = f"{backend_url.rstrip('/')}/health"
        print(f"\nTesting health endpoint at: {health_url}")

        response = requests.get(health_url, timeout=30)

        print(f"Health endpoint status: {response.status_code}")

        if response.status_code == 200:
            try:
                json_response = response.json()
                print(f"Health response: {json_response}")
                return True
            except:
                print(f"Health response (non-JSON): {response.text}")
                return response.status_code == 200
        else:
            print(f"Health endpoint returned status: {response.status_code}")
            print(f"Health response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Health endpoint test failed with error: {e}")
        return False

def main():
    """
    Main function to run the response test.
    """
    print("🔍 Testing Backend Response")
    print("=" * 50)

    # You'll need to replace this with your actual backend URL
    backend_url = input("Please enter your backend URL (e.g., https://your-space-name.hf.space): ").strip()

    if not backend_url:
        print("❌ No URL provided. Exiting.")
        sys.exit(1)

    # Test basic response
    basic_test_passed = test_backend_response(backend_url)

    # Test health endpoint
    health_test_passed = test_health_endpoint(backend_url)

    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"Basic endpoint test: {'✅ PASSED' if basic_test_passed else '❌ FAILED'}")
    print(f"Health endpoint test: {'✅ PASSED' if health_test_passed else '❌ FAILED'}")

    if basic_test_passed and health_test_passed:
        print("\n🎉 All tests passed! Your backend is responding correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Please check your backend deployment.")
        return False

if __name__ == "__main__":
    main()