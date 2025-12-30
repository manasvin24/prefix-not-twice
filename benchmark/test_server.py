"""
Simple test script to demonstrate the integrated prefix caching system.
"""

import requests
import time

SERVER_URL = "http://localhost:8000/generate"
CACHE_CLEAR_URL = "http://localhost:8000/cache/clear"
CACHE_STATS_URL = "http://localhost:8000/cache/stats"

def test_scenario(name, prompt, use_cache, use_prefix_cache):
    """Test a specific scenario."""
    print(f"\n{'='*60}")
    print(f"Scenario: {name}")
    print(f"{'='*60}")
    
    response = requests.post(SERVER_URL, json={
        "prompt": prompt,
        "max_new_tokens": 10,
        "use_cache": use_cache,
        "use_prefix_cache": use_prefix_cache
    })
    
    data = response.json()
    timings = data["timings"]
    
    print(f"Prompt: {prompt[:50]}..." if len(prompt) > 50 else f"Prompt: {prompt}")
    print(f"Use cache: {use_cache}, Use prefix cache: {use_prefix_cache}")
    print(f"\nTimings:")
    print(f"  Tokenize: {timings['tokenize_ms']:.1f}ms")
    print(f"  Cache lookup: {timings.get('cache_lookup_ms', 0):.1f}ms")
    print(f"  TTFT: {timings['ttft_ms']:.1f}ms")
    print(f"  Decode: {timings['decode_ms']:.1f}ms")
    print(f"  Total: {timings['total_ms']:.1f}ms")
    
    if "prefix_tokens_saved" in timings:
        print(f"  ✓ PREFIX CACHE HIT! Saved {timings['prefix_tokens_saved']} tokens")
    
    print(f"\nGenerated text: {data['text'][:100]}...")
    
    return timings

def main():
    print("\n" + "="*60)
    print("INTEGRATED PREFIX CACHING SYSTEM TEST")
    print("="*60)
    
    # Clear cache
    requests.post(CACHE_CLEAR_URL)
    print("\n✓ Cache cleared")
    
    # Test 1: Without any cache (baseline)
    print("\n" + "="*60)
    print("TEST 1: NO CACHE (Baseline)")
    print("="*60)
    t1 = test_scenario(
        "No cache",
        "What is artificial intelligence?",
        use_cache=False,
        use_prefix_cache=False
    )
    
    # Test 2: With KV-cache but no prefix cache
    print("\n" + "="*60)
    print("TEST 2: KV-CACHE ONLY")
    print("="*60)
    t2 = test_scenario(
        "KV-cache only",
        "What is machine learning?",
        use_cache=True,
        use_prefix_cache=False
    )
    
    # Test 3: First request with prefix cache (cache miss)
    print("\n" + "="*60)
    print("TEST 3: PREFIX CACHE - FIRST REQUEST (MISS)")
    print("="*60)
    requests.post(CACHE_CLEAR_URL)
    long_prompt = "You are a helpful AI assistant. Explain the following topic clearly and concisely: What is deep learning?"
    t3 = test_scenario(
        "Prefix cache - first request",
        long_prompt,
        use_cache=True,
        use_prefix_cache=True
    )
    
    time.sleep(0.5)
    
    # Test 4: Second request with same prompt (cache hit)
    print("\n" + "="*60)
    print("TEST 4: PREFIX CACHE - SECOND REQUEST (HIT)")
    print("="*60)
    t4 = test_scenario(
        "Prefix cache - cache hit",
        long_prompt,
        use_cache=True,
        use_prefix_cache=True
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY & ANALYSIS")
    print("="*60)
    
    # Get cache stats
    stats = requests.get(CACHE_STATS_URL).json()
    print(f"\nCache Statistics:")
    print(f"  Entries: {stats['cache_size']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    
    print(f"\nPerformance Comparison:")
    print(f"  No cache TTFT:        {t1['ttft_ms']:.1f}ms")
    print(f"  KV-cache TTFT:        {t2['ttft_ms']:.1f}ms")
    print(f"  Prefix cache miss:    {t3['ttft_ms']:.1f}ms")
    print(f"  Prefix cache hit:     {t4['ttft_ms']:.1f}ms")
    
    speedup = t3['ttft_ms'] / t4['ttft_ms'] if t4['ttft_ms'] > 0 else 0
    print(f"\n🚀 Prefix cache speedup: {speedup:.1f}x")
    print(f"   Saved {t3['ttft_ms'] - t4['ttft_ms']:.1f}ms on cache hit!")
    
    print("\n" + "="*60)
    print("✅ All tests completed successfully!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
