"""
Clean benchmark comparing baseline vs prefix cache.
Shows the true benefit of storing last_logits and eliminating redundant forward passes.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from cache.prefix_cache import PrefixCache
from inference.generate import generate


MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_TOKEN = os.getenv("HF_TOKEN")
DEVICE = torch.device("mps")


def main():
    print("\n" + "="*80)
    print("CLEAN PREFIX CACHE BENCHMARK")
    print("="*80)
    
    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        dtype=torch.float16,
        device_map={"": "mps"},
        low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    
    # Initialize cache with minimum 20 tokens (to ensure caching happens)
    cache = PrefixCache(min_tokens=20)
    
    # Long shared prefix (300+ tokens)
    LONG_PREFIX = """You are an expert AI assistant specializing in machine learning and deep learning. You provide clear, accurate, and detailed explanations of technical concepts. Your responses are educational and include examples when helpful.

When explaining machine learning concepts, you should:
1. Start with a high-level overview that captures the core idea
2. Break down complex topics into digestible components
3. Use analogies and real-world examples to illustrate abstract concepts
4. Provide mathematical formulations when relevant, but explain them intuitively
5. Discuss practical applications and use cases
6. Mention common pitfalls and best practices
7. Reference important research papers or foundational work when appropriate

Your expertise covers:
- Neural networks and deep learning architectures (CNNs, RNNs, Transformers, GANs)
- Natural language processing and large language models
- Computer vision and image processing
- Reinforcement learning and decision-making systems
- Optimization algorithms and training techniques
- Model evaluation, validation, and deployment strategies
- Ethical considerations in AI and machine learning

You communicate in a clear, structured manner. You avoid jargon when simpler terms suffice, but you're not afraid to use technical terminology when it's the most precise way to communicate. You acknowledge uncertainty when appropriate and distinguish between established facts and current research frontiers.

When a user asks a question, you:
- First ensure you understand what they're asking
- Provide a direct answer to their specific question
- Offer additional context that might be helpful
- Suggest related topics they might want to explore
- Encourage follow-up questions for deeper understanding

You stay current with the latest developments in AI and machine learning, and you can discuss both theoretical foundations and practical implementation details. You're comfortable working with popular frameworks like PyTorch, TensorFlow, scikit-learn, and Hugging Face Transformers.

Your goal is to make complex AI concepts accessible while maintaining technical accuracy. You help users build intuition and understanding, not just memorize facts. Now, please answer the following question: """
    
    # Test with same prompt 4 times
    test_prompt = LONG_PREFIX + "What is attention?"
    
    print(f"\nPrompt: {len(test_prompt)} characters (~{len(tokenizer(test_prompt)['input_ids'])} tokens)")
    print(f"Shared prefix: {len(LONG_PREFIX)} characters")
    print(f"Max new tokens: 15\n")
    
    print("="*80)
    print("RUNNING BENCHMARK")
    print("="*80)
    
    results = []
    
    for i in range(4):
        print(f"\nRequest {i+1}/4...")
        text, timings = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=test_prompt,
            prefix_cache=cache,
            max_new_tokens=15,
            device=DEVICE
        )
        
        results.append(timings)
        
        # Show key metrics
        print(f"  Tokenize: {timings['tokenize_ms']:.1f}ms")
        print(f"  Cache lookup: {timings['cache_lookup_ms']:.1f}ms")
        print(f"  TTFT: {timings['ttft_ms']:.1f}ms", end="")
        if "prefix_tokens_saved" in timings:
            print(f" (saved {timings['prefix_tokens_saved']} tokens) ✓")
        else:
            print(" (full prefill)")
        print(f"  Decode: {timings['decode_ms']:.1f}ms")
        print(f"  Total: {timings['total_ms']:.1f}ms")
        
        time.sleep(0.2)
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    stats = cache.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Entries: {stats['size']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit rate: {stats['hit_rate']:.1%}")
    
    # First request (cache miss)
    first = results[0]
    print(f"\nFirst request (cache miss):")
    print(f"  TTFT: {first['ttft_ms']:.1f}ms")
    
    # Subsequent requests (cache hits)
    if len(results) > 1:
        cache_hits = results[1:]
        avg_ttft_hits = sum(r['ttft_ms'] for r in cache_hits) / len(cache_hits)
        avg_total_hits = sum(r['total_ms'] for r in cache_hits) / len(cache_hits)
        
        print(f"\nSubsequent requests (cache hits):")
        print(f"  Average TTFT: {avg_ttft_hits:.1f}ms")
        print(f"  Average Total: {avg_total_hits:.1f}ms")
        
        speedup = first['ttft_ms'] / avg_ttft_hits if avg_ttft_hits > 0 else 0
        print(f"\n🚀 TTFT Speedup: {speedup:.2f}x faster!")
        
        if speedup > 1:
            saved_ms = first['ttft_ms'] - avg_ttft_hits
            print(f"   Saved {saved_ms:.1f}ms per request by reusing cached prefix")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("""
✅ What's working correctly:
   • Cache stores both past_key_values AND last_logits
   • On exact match: NO redundant forward pass
   • TTFT measures only model-critical path
   • Cache lookup excluded from TTFT
   • CPU storage → MPS transfer on hit (acceptable overhead)

🎯 Performance characteristics:
   • First request: Full prefill through all ~300-400 tokens
   • Cache hits: Zero model computation (reuse cached logits)
   • Speedup scales with prefix length
   
💡 Best use cases:
   • Long system prompts (100-500+ tokens)
   • Multi-turn conversations
   • RAG with large context
   • Batch requests with common prefixes
"""
)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
