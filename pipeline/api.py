"""
pipeline/api.py
DarkSyntax — FastAPI Endpoint
Person B owns this file.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from pipeline.triage import run_triage

app = FastAPI()

class TriageRequest(BaseModel):
    symptoms: str
    doc_path: str = "data/mtsamples.txt"

@app.post("/triage")
def triage(request: TriageRequest):
    result = run_triage(
        patient_input = request.symptoms,
        doc_path      = request.doc_path
    )
    return result

@app.get("/health")
def health():
    return {"status": "ok"}