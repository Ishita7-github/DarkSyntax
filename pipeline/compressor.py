"""
pipeline/compressor.py
DarkSyntax — ScaleDown Compression Module
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

SCALEDOWN_URL = "https://api.scaledown.xyz/compress/raw/"

def compress_for_query(query: str, doc_path: str, top_k: int = 15) -> dict:

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"Document not found: {doc_path}")

    start   = time.time()
    context = open(doc_path, encoding="utf-8").read()

    response = requests.post(
        SCALEDOWN_URL,
        headers={
            "x-api-key":    os.getenv("SCALEDOWN_API_KEY"),
            "Content-Type": "application/json"
        },
        json={
            "context":   context,
            "prompt":    query,
            "scaledown": {"rate": "auto"}
        },
        timeout=10
    )

    result            = response.json()
    compressed        = result.get("compressed_prompt", context[:500])
    original_tokens   = len(context.split())
    compressed_tokens = len(compressed.split())
    ratio = (
        round(original_tokens / compressed_tokens, 2)
        if compressed_tokens > 0 else 0.0
    )

    return {
        "compressed_context": compressed,
        "original_tokens":    original_tokens,
        "compressed_tokens":  compressed_tokens,
        "compression_ratio":  ratio,
        "latency_ms":         round((time.time() - start) * 1000),
    }

def compress_batch(queries: list, doc_path: str) -> list:
    return [compress_for_query(q, doc_path) for q in queries]