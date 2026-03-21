"""
test_pipeline.py
DarkSyntax — Full pipeline smoke test
Run from repo root: python test_pipeline.py

Tests all 3 pipeline files together:
  ner.py → extractor.py → compressor.py → triage.py

Prints a clean pass/fail table.
"""

import os
import sys
import time

PASS = " PASS "
FAIL = " FAIL "
SKIP = " SKIP "

results = []

def check(label, fn):
    try:
        val = fn()
        results.append((PASS, label, ""))
        print(f"  [{PASS}] {label}")
        return val
    except Exception as e:
        results.append((FAIL, label, str(e)))
        print(f"  [{FAIL}] {label}")
        print(f"          {e}")
        return None

def skip(label, reason):
    results.append((SKIP, label, reason))
    print(f"  [{SKIP}] {label} — {reason}")


print("\n" + "=" * 60)
print("DarkSyntax — Pipeline Smoke Test")
print("=" * 60)

# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n[1] Imports")

check("import spacy",           lambda: __import__("spacy"))
check("import scispacy",        lambda: __import__("scispacy"))
check("import fitz (PyMuPDF)",  lambda: __import__("fitz"))
check("import scaledown",       lambda: __import__("scaledown"))
check("import anthropic",       lambda: __import__("anthropic"))
check("import dotenv",          lambda: __import__("dotenv"))
check("load en_ner_bc5cdr_md",  lambda: __import__("spacy").load("en_ner_bc5cdr_md"))

# ── 2. NER module ─────────────────────────────────────────────────────────────
print("\n[2] NER module (pipeline/ner.py)")

from pipeline.ner import extract_entities

ner_result = check(
    "extract_entities runs",
    lambda: extract_entities(
        "severe chest pain radiating to left arm, BP 90/60, HR 110 bpm"
    )
)

if ner_result:
    check("risk_level is critical",
          lambda: None if ner_result["risk_level"] == "critical"
          else (_ for _ in ()).throw(AssertionError(
              f"Expected critical, got {ner_result['risk_level']}"
          )))

    check("query string not empty",
          lambda: None if ner_result["query"]
          else (_ for _ in ()).throw(AssertionError("Query is empty")))

    check("vitals extracted (blood_pressure)",
          lambda: None if "blood_pressure" in ner_result["vitals"]
          else (_ for _ in ()).throw(AssertionError(
              f"Vitals: {ner_result['vitals']}"
          )))

    print(f"\n    risk_level : {ner_result['risk_level']}")
    print(f"    symptoms   : {ner_result['symptoms']}")
    print(f"    vitals     : {ner_result['vitals']}")
    print(f"    query      : {ner_result['query'][:60]}")

# ── 3. Extractor module ───────────────────────────────────────────────────────
print("\n[3] Extractor module (pipeline/extractor.py)")

from pipeline.extractor import chunk_document, load_document

check("chunk_document works",
      lambda: chunk_document("word " * 500, chunk_size=100, overlap=20))

chunks = chunk_document("word " * 500, chunk_size=100, overlap=20)
check("chunks produced",
      lambda: None if len(chunks) > 0
      else (_ for _ in ()).throw(AssertionError("No chunks produced")))

DOC = "data/mtsamples.txt"
if os.path.exists(DOC):
    check("load_document reads corpus",
          lambda: load_document(DOC))
    text = load_document(DOC)
    print(f"\n    corpus size: {len(text.split()):,} words")
else:
    skip("load_document corpus", f"{DOC} not found — run extractor.py first")

# ── 4. Compressor module ──────────────────────────────────────────────────────
print("\n[4] Compressor module (pipeline/compressor.py)")

from dotenv import load_dotenv
load_dotenv()

if not os.getenv("SCALEDOWN_API_KEY"):
    skip("ScaleDown compress", "SCALEDOWN_API_KEY not set in .env")
elif not os.path.exists(DOC):
    skip("ScaleDown compress", f"{DOC} not found")
else:
    from pipeline.compressor import compress_for_query

    print("    Compressing (this takes ~2-3 seconds) ...")
    compressed = check(
        "compress_for_query runs",
        lambda: compress_for_query(
            "chest pain difficulty breathing",
            DOC,
            top_k=5,
        )
    )

    if compressed:
        check("compression_ratio > 1",
              lambda: None if compressed["compression_ratio"] > 1
              else (_ for _ in ()).throw(AssertionError(
                  f"Ratio: {compressed['compression_ratio']}"
              )))

        check("compressed_context not empty",
              lambda: None if compressed["compressed_context"]
              else (_ for _ in ()).throw(AssertionError("Empty context")))

        print(f"\n    original  : {compressed['original_tokens']:,} tokens")
        print(f"    compressed: {compressed['compressed_tokens']} tokens")
        print(f"    ratio     : {compressed['compression_ratio']}x")
        print(f"    latency   : {compressed['latency_ms']}ms")

# ── 5. Full triage pipeline ───────────────────────────────────────────────────
print("\n[5] Full triage pipeline (pipeline/triage.py)")

if not os.getenv("LLM_API_KEY"):
    skip("run_triage", "LLM_API_KEY not set in .env")
elif not os.path.exists(DOC):
    skip("run_triage", f"{DOC} not found")
else:
    from pipeline.triage import run_triage

    print("    Running full pipeline (10-15 seconds) ...")
    start = time.time()

    triage = check(
        "run_triage returns result",
        lambda: run_triage(
            "Severe chest pain radiating to left arm, sweating, "
            "difficulty breathing. BP 90/60.",
            DOC
        )
    )

    if triage:
        check("urgency field present",
              lambda: None if triage.get("urgency")
              else (_ for _ in ()).throw(AssertionError("No urgency field")))

        check("confidence is float",
              lambda: None if isinstance(triage.get("confidence"), float)
              else (_ for _ in ()).throw(AssertionError(
                  f"confidence type: {type(triage.get('confidence'))}"
              )))

        check("latency_ms present",
              lambda: None if triage.get("latency_ms")
              else (_ for _ in ()).throw(AssertionError("No latency_ms")))

        print(f"\n    diagnosis  : {triage['diagnosis']}")
        print(f"    urgency    : {triage['urgency'].upper()}")
        print(f"    confidence : {triage['confidence']}")
        print(f"    latency    : {triage['latency_ms']}ms")
        print(f"    ratio      : {triage['compression_ratio']}x compression")
        print(f"    action     : {triage['recommended_action']}")

        if triage["latency_ms"] > 500:
            print(f"\n    [WARN] {triage['latency_ms']}ms exceeds 500ms target")
            print("    Try: run_triage(text, doc, top_k=8)")

# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
skipped = sum(1 for r in results if r[0] == SKIP)

print("\n" + "=" * 60)
print(f"  {passed} passed   {failed} failed   {skipped} skipped")

if failed == 0:
    print("\n  Pipeline ready. Handoff to Person B:")
    print("    from pipeline.triage import run_triage")
    print("    result = run_triage(patient_text, 'data/mtsamples.txt')\n")
    sys.exit(0)
else:
    print("\n  Fix FAIL items before handing off to Person B.\n")
    sys.exit(1)
