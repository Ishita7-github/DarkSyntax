
# 🚨 DarkSyntax — Emergency Response Triage Assistant

AI-powered emergency triage pipeline built by three students as a learning project.

## What It Does

A doctor types patient symptoms, uploads the patient history PDF and hospital protocol PDF, and gets a colour-coded triage result. The system extracts named entities from the symptoms, compresses the patient PDF using the ScaleDown API to only the most relevant chunks, and calls Groq (Llama 3.3 70B) to generate a structured diagnosis with urgency level, recommended action, medication warnings, and cited records.

## How to Use

1. Type the patient's symptoms in the text box
2. Upload the Patient History PDF
3. Upload the Hospital Protocol PDF
4. Click **Run Triage**

## Pipeline Architecture
```
Symptoms + PDFs
      ↓
NER extraction (spaCy + scispaCy)
      ↓
ScaleDown API compression
(relevance-based token reduction)
      ↓
Groq LLM — Llama 3.3 70B
      ↓
Confidence scoring + audit log
      ↓
Colour coded result
```

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** FastAPI
- **LLM:** Groq API (Llama 3.3 70B)
- **NER:** spaCy + scispaCy (en_ner_bc5cdr_md)
- **Compression:** ScaleDown API
- **PDF reading:** PyMuPDF

## Team

| Name | Role |
|------|------|
| Ishita Bhatt | ML Lead — PDF processing, ScaleDown compression pipeline |
| Harsh Jhunjhunwala | Backend + LLM — FastAPI, Groq integration |
| Priyanshu Kr Gupta | Frontend + Safety — Streamlit UI, confidence scoring, audit logs |

## Environment Variables Required

| Key | Description |
|-----|-------------|
| `LLM_API_KEY` | Your Groq API key from console.groq.com |
| `SCALEDOWN_API_KEY` | Your ScaleDown API key from scaledown.xyz |

