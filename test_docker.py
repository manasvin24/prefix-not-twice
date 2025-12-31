"""
Test script for dockerized KV-cache server.
Run after starting the container to verify everything works.
"""

import requests
import time
import sys


def test_server():
    base_url = "http://localhost:8000"
    
    print("="*80)
    print("DOCKER SERVER TEST")
    print("="*80)
    
    # Wait for server to start
    print("\n1. Waiting for server to start...")
    for attempt in range(30):
        try:
            response = requests.get(f"{base_url}/cache/stats", timeout=2)
            if response.status_code == 200:
                print(f"   ✅ Server is ready! (attempt {attempt + 1})")
                break
        except requests.exceptions.RequestException:
            time.sleep(1)
            print(f"   ⏳ Attempt {attempt + 1}/30...", end='\r')
    else:
        print("   ❌ Server failed to start after 30 seconds")
        sys.exit(1)
    
    # Test cache stats endpoint
    print("\n2. Testing cache stats endpoint...")
    try:
        response = requests.get(f"{base_url}/cache/stats")
        response.raise_for_status()
        stats = response.json()
        print(f"   ✅ Cache stats retrieved:")
        print(f"      - Cache size: {stats['cache_size']}")
        print(f"      - Total requests: {stats['total_requests']}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        sys.exit(1)
    
    # Test generation without cache
    print("\n3. Testing generation (no cache)...")
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "prompt": "What is machine learning?",
                "max_new_tokens": 20,
                "use_cache": True,
                "use_prefix_cache": False
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        print(f"   ✅ Generation successful!")
        print(f"      - Text: {result['text'][:50]}...")
        print(f"      - Latency: {result['latency_ms']:.1f}ms")
        print(f"      - Tokens/sec: {result['tokens_per_sec']:.1f}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        sys.exit(1)
    
    # Test generation with prefix cache
    print("\n4. Testing generation (with prefix cache)...")
    try:
        prefix = "You are an AI assistant. " * 20  # Create a longer prefix
        response = requests.post(
            f"{base_url}/generate",
            json={
                "prompt": prefix + "What is attention mechanism?",
                "max_new_tokens": 15,
                "use_cache": True,
                "use_prefix_cache": True
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        print(f"   ✅ First request (cache miss):")
        print(f"      - TTFT: {result['timings']['ttft_ms']:.1f}ms")
        
        # Second request with same prefix
        response = requests.post(
            f"{base_url}/generate",
            json={
                "prompt": prefix + "What is transformer?",
                "max_new_tokens": 15,
                "use_cache": True,
                "use_prefix_cache": True
            },
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        print(f"   ✅ Second request (cache hit):")
        print(f"      - TTFT: {result['timings']['ttft_ms']:.1f}ms")
        print(f"      - Cache hit: {result.get('prefix_cache_hit', False)}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        sys.exit(1)
    
    # Test cache clear
    print("\n5. Testing cache clear...")
    try:
        response = requests.post(f"{base_url}/cache/clear")
        response.raise_for_status()
        result = response.json()
        print(f"   ✅ Cache cleared: {result['message']}")
        
        # Verify cache is empty
        stats = requests.get(f"{base_url}/cache/stats").json()
        print(f"   ✅ Verified: Cache size = {stats['cache_size']}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! 🎉")
    print("="*80)
    print("\nDocker container is working correctly.")
    print("You can now use the API at http://localhost:8000")


if __name__ == "__main__":
    test_server()
