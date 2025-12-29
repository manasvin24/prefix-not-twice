import asyncio
import httpx
import time
import numpy as np

SERVER_URL = "http://localhost:8000/generate"
PROMPT = "Explain KV caching in one sentence."
MAX_NEW_TOKENS = 32
N_CONCURRENT_REQUESTS = 10

async def call_api(prompt, use_cache=True):
    async with httpx.AsyncClient(timeout=120.0) as client:  # Increased timeout
        start = time.time()
        response = await client.post(
            SERVER_URL,
            json={
                "prompt": prompt,
                "max_new_tokens": MAX_NEW_TOKENS,
                "use_cache": use_cache
            }
        )
        end = time.time()
        latency = end - start
        return latency, response.json()

async def run_benchmark(use_cache=True, n_requests=N_CONCURRENT_REQUESTS):
    """Run benchmark with specified cache mode."""
    print(f"\n{'='*60}")
    print(f"Running benchmark: {'WITH' if use_cache else 'WITHOUT'} KV-cache")
    print(f"Concurrent requests: {n_requests}")
    print(f"{'='*60}")
    
    # Fire N concurrent requests with repeated prefixes
    tasks = [call_api(PROMPT, use_cache=use_cache) for _ in range(n_requests)]
    results = await asyncio.gather(*tasks)

    latencies = [r[0] for r in results]
    responses = [r[1] for r in results]
    
    # Extract server-side metrics
    server_latencies = [r.get('latency_ms', 0) for r in responses]
    tokens_per_sec = [r.get('tokens_per_sec', 0) for r in responses]
    
    print(f"\n--- Client-side Latency (includes network) ---")
    print(f"p50 latency: {np.percentile(latencies, 50)*1000:.2f} ms")
    print(f"p95 latency: {np.percentile(latencies, 95)*1000:.2f} ms")
    print(f"Average latency: {np.mean(latencies)*1000:.2f} ms")
    print(f"Min latency: {np.min(latencies)*1000:.2f} ms")
    print(f"Max latency: {np.max(latencies)*1000:.2f} ms")
    
    print(f"\n--- Server-side Latency (generation only) ---")
    print(f"p50 latency: {np.percentile(server_latencies, 50):.2f} ms")
    print(f"p95 latency: {np.percentile(server_latencies, 95):.2f} ms")
    print(f"Average latency: {np.mean(server_latencies):.2f} ms")
    
    print(f"\n--- Throughput ---")
    print(f"Total time: {np.sum(latencies):.2f} seconds")
    print(f"Throughput: {n_requests/np.sum(latencies):.2f} requests/sec")
    print(f"Average tokens/sec: {np.mean(tokens_per_sec):.2f}")
    print(f"{'='*60}\n")
    
    return {
        'latencies': latencies,
        'server_latencies': server_latencies,
        'tokens_per_sec': tokens_per_sec,
        'use_cache': use_cache
    }

async def main():
    """Run comparison benchmark: with and without cache."""
    print("\n" + "="*60)
    print("KV-CACHE INFERENCE BENCHMARK")
    print("="*60)
    
    # Test with cache
    results_with_cache = await run_benchmark(use_cache=True, n_requests=N_CONCURRENT_REQUESTS)
    
    # Wait a bit between tests
    await asyncio.sleep(2)
    
    # Test without cache
    results_without_cache = await run_benchmark(use_cache=False, n_requests=N_CONCURRENT_REQUESTS)
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    speedup = np.mean(results_without_cache['server_latencies']) / np.mean(results_with_cache['server_latencies'])
    throughput_improvement = np.mean(results_with_cache['tokens_per_sec']) / np.mean(results_without_cache['tokens_per_sec'])
    
    print(f"Average latency WITH cache: {np.mean(results_with_cache['server_latencies']):.2f} ms")
    print(f"Average latency WITHOUT cache: {np.mean(results_without_cache['server_latencies']):.2f} ms")
    print(f"Speedup: {speedup:.2f}x faster with cache")
    print(f"Throughput improvement: {throughput_improvement:.2f}x")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
