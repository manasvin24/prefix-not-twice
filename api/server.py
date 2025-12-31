from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

from cache.prefix_cache import PrefixCache
from inference.generate import generate as clean_generate

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_TOKEN = os.getenv("HF_TOKEN")

# Auto-detect device: MPS (Mac) > CUDA (GPU) > CPU
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float16
    print("🚀 Using Metal Performance Shaders (MPS)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
    print("🚀 Using CUDA GPU")
else:
    DEVICE = "cpu"
    DTYPE = torch.float32
    print("⚠️  Using CPU (slower, but works everywhere)")

app = FastAPI()

# Load once at startup
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=DTYPE,
    device_map={"": DEVICE},
    low_cpu_mem_usage=True
)
model.eval()
print(f"✅ Model loaded on {DEVICE}")

# Initialize prefix cache (min 20 tokens for meaningful caching)
prefix_cache = PrefixCache(min_tokens=20)

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    use_cache: bool = True  # Toggle KV-cache on/off
    use_prefix_cache: bool = True  # Toggle prefix caching

class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    tokens_per_sec: float
    timings: dict
    cache_enabled: bool
    prefix_cache_hit: bool

@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    """
    Generate text with optional KV-cache and prefix caching.
    
    The 8-step inference flow:
    1. Tokenize prompt
    2. Cache lookup (if prefix_cache enabled)
    3. Split into cached prefix + suffix
    4. On hit: Load past_key_values, run model on suffix only
    5. On miss: Full prefill, save KV states + logits
    6. Start decode loop with cached/computed logits
    7. Generate tokens using KV-cache
    8. Return generated text with timing metrics
    """
    
    # Use clean generate implementation
    # Pass a dummy cache with impossible min_tokens to disable prefix caching
    active_cache = prefix_cache if req.use_prefix_cache else PrefixCache(min_tokens=999999)
    
    text, timings = clean_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=req.prompt,
        prefix_cache=active_cache,
        max_new_tokens=req.max_new_tokens,
        use_cache=req.use_cache,
        device=torch.device(DEVICE)
    )
    
    # Calculate metrics
    total_ms = timings["total_ms"]
    tokens_generated = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    tokens_per_sec = (tokens_generated / total_ms * 1000) if total_ms > 0 else 0
    prefix_cache_hit = "prefix_tokens_saved" in timings
    
    # Console output
    cache_status = "WITH CACHE" if req.use_cache else "WITHOUT CACHE"
    if prefix_cache_hit:
        cache_status += " + PREFIX CACHE HIT"
    
    print(f"\n{'='*50}")
    print(f"[MODE] {cache_status}")
    if prefix_cache_hit:
        print(f"[Prefix Cache] HIT! Saved {timings['prefix_tokens_saved']} tokens")
        stats = prefix_cache.get_stats()
        print(f"[Prefix Cache Stats] Hit rate: {stats['hit_rate']:.2%} ({stats['hits']}/{stats['total_requests']})")
    print(f"[Tokenize] {timings.get('tokenize_ms', 0):.2f} ms")
    print(f"[Cache Lookup] {timings.get('cache_lookup_ms', 0):.2f} ms")
    print(f"[TTFT] {timings['ttft_ms']:.2f} ms (Time To First Token)")
    print(f"[Decode] {timings['decode_ms']:.2f} ms")
    print(f"[Total] {total_ms:.2f} ms")
    print(f"[Throughput] {tokens_per_sec:.2f} tokens/sec")
    print(f"{'='*50}\n")

    return GenerateResponse(
        text=text,
        latency_ms=total_ms,
        tokens_per_sec=tokens_per_sec,
        cache_enabled=req.use_cache,
        prefix_cache_hit=prefix_cache_hit,
        timings=timings
    )

@app.get("/cache/stats")
def get_cache_stats():
    """Get prefix cache statistics."""
    stats = prefix_cache.get_stats()
    return {
        "cache_size": stats["size"],
        "hits": stats["hits"],
        "misses": stats["misses"],
        "hit_rate": stats["hit_rate"],
        "total_requests": stats["total_requests"]
    }

@app.post("/cache/clear")
def clear_cache():
    """Clear the prefix cache."""
    prefix_cache.clear()
    return {
        "status": "success",
        "message": "Prefix cache cleared"
    }
