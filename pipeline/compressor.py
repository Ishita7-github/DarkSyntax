"""
pipeline/compressor.py
DarkSyntax — ScaleDown Compression Module
Person A owns this file.

Uses ScaleDown API to compress the 200-page document corpus
down to ~200 tokens relevant to the patient query.

Versions: scaledown[semantic,haste]>=0.1.0
"""

import os
import time
from dotenv import load_dotenv
from scaledown import ScaleDownCompressor, Pipeline
from scaledown.optimizer import SemanticOptimizer

load_dotenv()


# ─── Pipeline builder ─────────────────────────────────────────────────────────

def build_compression_pipeline(top_k: int = 15) -> Pipeline:
    """
    Builds a two-stage ScaleDown pipeline:
      Stage 1 — SemanticOptimizer: retrieves top_k relevant chunks
      Stage 2 — ScaleDownCompressor: compresses to token budget

    top_k: number of chunks to retrieve before compression
           increase for better recall, decrease for lower latency
    """
    return Pipeline([
        ("retrieve", SemanticOptimizer(
            top_k=top_k,
            target_model="claude-haiku-4-5-20251001",
        )),
        ("compress", ScaleDownCompressor(
            target_model="claude-haiku-4-5-20251001",
            rate="auto",
            api_key=os.getenv("SCALEDOWN_API_KEY"),
        )),
    ])


# ─── Main compression function ────────────────────────────────────────────────

def compress_for_query(
    query:     str,
    doc_path:  str,
    top_k:     int = 15,
) -> dict:
    """
    Entry point called by triage.py.

    Takes a patient symptom query and a document path,
    retrieves relevant passages and compresses them.

    Returns:
    {
        "compressed_context": str,   ← ready to inject into LLM prompt
        "original_tokens":    int,   ← tokens before compression
        "compressed_tokens":  int,   ← tokens after compression
        "compression_ratio":  float, ← e.g. 12.4 means 12.4x reduction
        "latency_ms":         int,   ← time taken for this step
    }
    """
    if not os.path.exists(doc_path):
        raise FileNotFoundError(
            f"Document not found: {doc_path}\n"
            "Run extractor.py first to build the corpus."
        )

    if not os.getenv("SCALEDOWN_API_KEY"):
        raise ValueError(
            "SCALEDOWN_API_KEY not set.\n"
            "Add it to your .env file: SCALEDOWN_API_KEY=sk-..."
        )

    start    = time.time()
    pipeline = build_compression_pipeline(top_k=top_k)

    result = pipeline.run(
        query     = query,
        file_path = doc_path,
        prompt    = f"Find medical information relevant to: {query}",
    )

    latency = round((time.time() - start) * 1000)

    original_tokens   = getattr(result.metrics, "original_tokens",   0)
    compressed_tokens = getattr(result.metrics, "total_tokens",      0)

    # Guard against division by zero
    ratio = (
        round(original_tokens / compressed_tokens, 2)
        if compressed_tokens > 0
        else 0.0
    )

    return {
        "compressed_context": result.final_content,
        "original_tokens":    original_tokens,
        "compressed_tokens":  compressed_tokens,
        "compression_ratio":  ratio,
        "latency_ms":         latency,
    }


# ─── Batch compression (for self-consistency voting) ─────────────────────────

def compress_batch(
    queries:  list,
    doc_path: str,
) -> list:
    """
    Compress same document for multiple queries in parallel.
    Used by triage.py when running self-consistency voting (3 LLM runs).

    Returns list of compression result dicts in same order as queries.
    """
    compressor = ScaleDownCompressor(
        target_model = "claude-haiku-4-5-20251001",
        rate         = "auto",
        api_key      = os.getenv("SCALEDOWN_API_KEY"),
    )

    # ScaleDown supports batch mode natively
    results = compressor.compress(
        context = [open(doc_path).read()] * len(queries),
        prompt  = queries,
    )

    return [
        {
            "compressed_context": r.compressed_prompt,
            "original_tokens":    getattr(r.metrics, "original_prompt_tokens", 0),
            "compressed_tokens":  getattr(r.metrics, "compressed_prompt_tokens", 0),
        }
        for r in (results if isinstance(results, list) else [results])
    ]


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("DarkSyntax — Compressor Self-Test")
    print("=" * 55)

    # Check API key
    if not os.getenv("SCALEDOWN_API_KEY"):
        print("\n[FAIL] SCALEDOWN_API_KEY not found in .env")
        print("  Add: SCALEDOWN_API_KEY=sk-your-key")
        sys.exit(1)

    doc_path = "data/mtsamples.txt"
    if not os.path.exists(doc_path):
        print(f"\n[SKIP] {doc_path} not found.")
        print("  Run: python pipeline/extractor.py first")
        sys.exit(0)

    query = "severe chest pain radiating to left arm difficulty breathing"
    print(f"\nQuery    : {query}")
    print(f"Document : {doc_path}")
    print("Compressing ...")

    result = compress_for_query(query, doc_path)

    print(f"\nOriginal tokens   : {result['original_tokens']:,}")
    print(f"Compressed tokens : {result['compressed_tokens']:,}")
    print(f"Compression ratio : {result['compression_ratio']}x")
    print(f"Latency           : {result['latency_ms']}ms")
    print(f"\nCompressed context preview:")
    print("-" * 40)
    print(result["compressed_context"][:500])
    print("-" * 40)

    if result["latency_ms"] > 500:
        print(f"\n[WARN] Latency {result['latency_ms']}ms > 500ms target.")
        print("  Try reducing top_k: compress_for_query(query, path, top_k=8)")
    else:
        print(f"\n[OK] Within 500ms budget.")
