import streamlit as st
import time
import json
import fitz  # PyMuPDF
import re
from section_c import run_section_c

st.set_page_config(
    page_title='Emergency Triage Assistant',
    page_icon='🚨',
    layout='wide'
)

# ── HEADER ───────────────────────────────────────────
st.title('🚨 Emergency Response Triage Assistant')
st.caption('AI-powered. Sub-500ms. Zero hallucination tolerance.')
st.divider()

# ── HELPER: Read PDF ─────────────────────────────────
def read_pdf(uploaded_file) -> str:
    '''Extract text from uploaded PDF file.'''
    doc = fitz.open(stream=uploaded_file.read(), filetype='pdf')
    text = ''
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# ── HELPER: Clean Text ───────────────────────────────
def clean_text(text: str) -> str:
    '''Remove noise from PDF text.'''
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(line for line in lines if line).strip()

# ── INPUT SECTION ────────────────────────────────────
st.subheader('🩺 Patient Input')

symptoms = st.text_area(
    'Describe the emergency:',
    placeholder='62M, chest pain, left arm numbness, shortness of breath',
    height=120
)

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    history_file = st.file_uploader(
        '📄 Upload Patient History PDF',
        type=['pdf'],
        key='history'
    )
    if history_file:
        st.success(f'✅ Loaded: {history_file.name}')

with col_upload2:
    protocol_file = st.file_uploader(
        '📄 Upload Hospital Protocol PDF',
        type=['pdf'],
        key='protocol'
    )
    if protocol_file:
        st.success(f'✅ Loaded: {protocol_file.name}')

st.divider()

# ── RUN BUTTON ───────────────────────────────────────
if st.button('🚨 Run Triage', type='primary', use_container_width=True):

    # Validate inputs
    if not symptoms:
        st.error('⚠️ Please enter symptoms first.')
    elif not history_file:
        st.error('⚠️ Please upload patient history PDF.')
    elif not protocol_file:
        st.error('⚠️ Please upload hospital protocol PDF.')

    else:
        pipeline_start = time.time()

        # ── READ + CLEAN PDFs ────────────────────────
        with st.spinner('📄 Reading PDFs...'):
            raw_history  = read_pdf(history_file)
            raw_protocol = read_pdf(protocol_file)
            clean_history  = clean_text(raw_history)
            clean_protocol = clean_text(raw_protocol)

        # ── BUILD MOCK B OUTPUT WITH REAL PDF DATA ───
        # Replace this block when Person B gives you section_b.py
        mock_b_output = {
            'symptoms': symptoms,
            'diagnosis': 'Probable STEMI',
            'immediate_action': 'Activate cath lab. Administer aspirin 325mg. 12-lead ECG immediately.',
            'medications_to_check': ['Warfarin', 'Aspirin', 'Metoprolol'],
            'risk_level': 'HIGH',
            'cited_records': 'BP 145/90 from Jan 2024. DVT history 2022. Warfarin noted.',
            'warnings': 'Patient on Warfarin — bleeding risk. Penicillin allergy.',
            'context_used': clean_history[:500],  # real PDF data used here
            'llm_latency_ms': 210,
            'section_b_latency_ms': 215
        }

        # ── LIVE LATENCY COUNTER ─────────────────────
        latency_display = st.empty()
        status_display  = st.empty()

        with st.spinner(''):
            for i in range(20):
                elapsed = (time.time() - pipeline_start) * 1000
                latency_display.metric('⏱️ Pipeline Running...', f'{elapsed:.0f} ms')
                time.sleep(0.02)

            status_display.info('🟢 Running Section C — Verify + Score + Audit...')
            result = run_section_c(mock_b_output, total_pipeline_ms=450)

        total_ms = (time.time() - pipeline_start) * 1000

        # ── CLEAR LOADING STATE ──────────────────────
        latency_display.empty()
        status_display.empty()

        # ── LATENCY BANNER ───────────────────────────
        if total_ms < 500:
            st.success(f'✅ Total latency: {total_ms:.0f}ms — UNDER 500ms 🎯')
        else:
            st.error(f'❌ Total latency: {total_ms:.0f}ms — OVER 500ms')

        # ── CONFIDENCE BANNER ────────────────────────
        confidence = result['confidence']
        if confidence['requires_human_review']:
            st.error(f'🔴 Confidence: {confidence["score"]} — {confidence["status"]}')
        else:
            st.success(f'✅ Confidence: {confidence["score"]} — {confidence["status"]}')

        st.divider()

        # ── RESULT CARDS ─────────────────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader('🔍 Diagnosis')
            risk = result.get('risk_level', 'UNKNOWN')

            if risk == 'HIGH':
                st.error(f'🔴 {result.get("diagnosis", "N/A")}')
                st.error(f'Risk Level: {risk}')
            elif risk == 'MEDIUM':
                st.warning(f'🟠 {result.get("diagnosis", "N/A")}')
                st.warning(f'Risk Level: {risk}')
            else:
                st.success(f'🟢 {result.get("diagnosis", "N/A")}')
                st.success(f'Risk Level: {risk}')

        with col_b:
            st.subheader('⚡ Immediate Action')
            st.info(result.get('immediate_action', 'N/A'))

        st.divider()

        # ── CITED RECORDS + WARNINGS ─────────────────
        col_c, col_d = st.columns(2)

        with col_c:
            st.subheader('📋 Cited Records')
            st.write(result.get('cited_records', 'N/A'))

        with col_d:
            st.subheader('⚠️ Warnings')
            if result.get('warnings'):
                st.warning(result['warnings'])
            else:
                st.success('No warnings')

        st.divider()

        # ── MEDICATIONS ──────────────────────────────
        st.subheader('💊 Medications to Check')
        meds = result.get('medications_to_check', [])
        if meds:
            cols = st.columns(len(meds))
            for i, med in enumerate(meds):
                cols[i].error(f'💊 {med}')

        st.divider()

        # ── PDF PREVIEW ──────────────────────────────
        with st.expander('📄 Patient History Preview (first 500 chars)'):
            st.text(clean_history[:500])

        with st.expander('📄 Protocol Preview (first 500 chars)'):
            st.text(clean_protocol[:500])

        # ── AUDIT LOG ────────────────────────────────
        with st.expander('📁 Full Audit Log'):
            st.json(result)
            st.caption(f'Saved to: {result["audit_file"]}')