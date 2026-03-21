"""
pipeline/triage.py
DarkSyntax — Core Triage Engine
Person A owns this file.

Full pipeline:
  Patient text → NER → ScaleDown compression → LLM (CoT) → JSON output

Output schema (shared with Person B + C):
{
    "diagnosis":          str,
    "urgency":            "critical|high|moderate|low",
    "confidence":         float (0.0–1.0),
    "reasoning":          str,
    "red_flags":          list[str],
    "recommended_action": str,
    "negated_symptoms":   list[str],
    "medications":        list[str],
    "vitals":             dict,
    "latency_ms":         int,
    "original_tokens":    int,
    "compressed_tokens":  int,
    "compression_ratio":  float,
    "entities":           dict,
}
"""

import os
import json
import re
import time
from groq import Groq
from dotenv import load_dotenv

from pipeline.ner        import extract_entities
from pipeline.compressor import compress_for_query

load_dotenv()

# ─── LLM client ───────────────────────────────────────────────────────────────

def get_llm_client():
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise ValueError("LLM_API_KEY not set")
    return Groq(api_key=key)


# ─── Chain-of-Thought prompt ──────────────────────────────────────────────────

COT_PROMPT = """You are an experienced emergency triage physician assistant.
Analyze the patient information below and provide a structured triage assessment.

PATIENT INPUT:
{patient_input}

EXTRACTED ENTITIES:
- Symptoms/Diseases : {symptoms}
- Anatomy affected  : {anatomy}
- Medications       : {chemicals}
- Vitals            : {vitals}
- Severity flags    : {severity_flags}
- Denied symptoms   : {negations}
- Duration          : {duration_hints}

RELEVANT MEDICAL CONTEXT (compressed from hospital records):
{compressed_context}

Think step by step:
1. What are the key symptoms and their clinical significance?
2. Do the vitals indicate hemodynamic instability?
3. What is the most likely diagnosis given symptoms + context?
4. Are there any red flags requiring immediate intervention?
5. What is the urgency level?

Respond with ONLY this JSON — no extra text, no markdown, no code blocks:
{{
  "diagnosis": "primary diagnosis in plain English",
  "urgency": "critical or high or moderate or low",
  "confidence": 0.0,
  "reasoning": "2-3 sentence clinical reasoning",
  "red_flags": ["flag1", "flag2"],
  "recommended_action": "specific immediate action",
  "negated_symptoms": {negations},
  "medications": {chemicals},
  "vitals": {vitals}
}}

Rules:
- urgency must be exactly one of: critical, high, moderate, low
- confidence must be a float between 0.0 and 1.0
- red_flags must be a JSON array (can be empty [])
- Return ONLY the JSON object, nothing else"""


# ─── Safety pre-check ─────────────────────────────────────────────────────────

INSTANT_CRITICAL_KEYWORDS = [
    "not breathing", "no pulse", "unconscious", "unresponsive",
    "severe bleeding", "overdose", "seizure", "stroke",
    "anaphylaxis", "heart attack",
]


def _instant_critical_check(text: str) -> bool:
    """Returns True if text contains keywords requiring immediate critical alert."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in INSTANT_CRITICAL_KEYWORDS)


# ─── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str, client) -> dict:
    response = client.chat.completions.create(
        model      = "llama-3.1-8b-instant",
        max_tokens = 600,
        messages   = [{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
          return {
            "diagnosis":          "Unable to parse — see reasoning",
            "urgency":            "high",
            "confidence":         0.5,
            "reasoning":          raw[:300],
            "red_flags":          [],
            "recommended_action": "Escalate to physician immediately",
        }


# ─── Main triage function ─────────────────────────────────────────────────────

def run_triage(
    patient_input: str,
    doc_path:      str = "data/mtsamples.txt",
    top_k:         int = 15,
) -> dict:
    """
    Full triage pipeline. Called by Person B's FastAPI endpoint.

    Args:
        patient_input: raw symptom text from chatbot
        doc_path:      path to hospital document corpus
        top_k:         retrieval depth for ScaleDown

    Returns complete triage result dict matching agreed JSON schema.
    """
    import re  # imported here to avoid module-level dep for re in COT_PROMPT
    start_total = time.time()

    # ── Safety pre-check ──────────────────────────────────────────────────────
    if _instant_critical_check(patient_input):
        return {
            "diagnosis":          "Immediate life-threatening emergency",
            "urgency":            "critical",
            "confidence":         0.99,
            "reasoning":          "Auto-detected life-threatening keyword. "
                                  "Immediate physician intervention required.",
            "red_flags":          ["auto-detected critical keyword"],
            "recommended_action": "Call emergency services immediately (112/911)",
            "negated_symptoms":   [],
            "medications":        [],
            "vitals":             {},
            "latency_ms":         round((time.time() - start_total) * 1000),
            "original_tokens":    0,
            "compressed_tokens":  0,
            "compression_ratio":  0.0,
            "entities":           {},
        }

    # ── Step 1: NER ───────────────────────────────────────────────────────────
    ner_result = extract_entities(patient_input)
    query      = ner_result["query"] or patient_input[:200]

    # ── Step 2: ScaleDown compression ────────────────────────────────────────
    compressed = compress_for_query(query, doc_path, top_k=top_k)

    # ── Step 3: Build CoT prompt ──────────────────────────────────────────────
    prompt = COT_PROMPT.format(
        patient_input      = patient_input,
        symptoms           = ner_result["symptoms"]       or "none detected",
        anatomy            = ner_result["anatomy"]        or "none detected",
        chemicals          = json.dumps(ner_result["chemicals"]),
        vitals             = json.dumps(ner_result["vitals"]),
        severity_flags     = ner_result["severity_flags"] or "none",
        negations          = json.dumps(ner_result["negations"]),
        duration_hints     = ner_result["duration_hints"] or "none",
        compressed_context = compressed["compressed_context"],
    )

    # ── Step 4: LLM call ──────────────────────────────────────────────────────
    client     = get_llm_client()
    llm_result = _call_llm(prompt, client)

    # ── Step 5: Assemble final output ─────────────────────────────────────────
    total_latency = round((time.time() - start_total) * 1000)

    return {
        # LLM outputs
        "diagnosis":          llm_result.get("diagnosis",          "Unknown"),
        "urgency":            llm_result.get("urgency",            "high"),
        "confidence":         llm_result.get("confidence",         0.0),
        "reasoning":          llm_result.get("reasoning",          ""),
        "red_flags":          llm_result.get("red_flags",          []),
        "recommended_action": llm_result.get("recommended_action", ""),
        "negated_symptoms":   ner_result["negations"],
        "medications":        ner_result["chemicals"],
        "vitals":             ner_result["vitals"],
        # Metrics (displayed in Person C's UI)
        "latency_ms":         total_latency,
        "original_tokens":    compressed["original_tokens"],
        "compressed_tokens":  compressed["compressed_tokens"],
        "compression_ratio":  compressed["compression_ratio"],
        # Full NER output for debugging
        "entities":           ner_result,
    }


# ─── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("DarkSyntax — Triage Engine Self-Test")
    print("=" * 60)

    TEST_CASES = [
        "Severe crushing chest pain radiating to left arm, "
        "profuse sweating, difficulty breathing. BP 90/60, HR 110.",

        "Sudden worst headache of life, stiff neck, "
        "fever 102F, sensitivity to light.",

        "Mild stomach ache after eating, no fever, "
        "no vomiting, no diarrhea.",
    ]

    DOC = "data/mtsamples.txt"

    if not os.path.exists(DOC):
        print(f"\n[SKIP] {DOC} not found — run extractor.py first")
        sys.exit(0)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[Case {i}] {case[:65]}...")
        try:
            result = run_triage(case, DOC)
            print(f"  Diagnosis  : {result['diagnosis']}")
            print(f"  Urgency    : {result['urgency'].upper()}")
            print(f"  Confidence : {result['confidence']}")
            print(f"  Latency    : {result['latency_ms']}ms")
            print(f"  Tokens     : {result['original_tokens']:,}"
                  f" → {result['compressed_tokens']} "
                  f"({result['compression_ratio']}x)")
            print(f"  Red flags  : {result['red_flags']}")
            print(f"  Action     : {result['recommended_action']}")

            if result["latency_ms"] > 500:
                print(f"  [WARN] Over 500ms budget!")
            else:
                print(f"  [OK] Within latency budget.")

        except Exception as e:
            print(f"  [ERROR] {e}")

    print("\n" + "=" * 60)
    print("Copy run_triage() import path for Person B:")
    print("  from pipeline.triage import run_triage")
    print("=" * 60)
