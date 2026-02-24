# app/streamlit_app.py — Streamlit Frontend
import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical ICD-10 Coder",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    }

    .hero-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.5rem;
        font-weight: 600;
        color: #00d4ff;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #8892a4;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #00d4ff;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }

    .stTextArea textarea {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        color: #e0e6f0 !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        border-radius: 10px !important;
    }

    .stButton button {
        background: linear-gradient(135deg, #00d4ff, #0099cc) !important;
        color: #0a0f1e !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: none !important;
        border-radius: 8px !important;
        width: 100% !important;
        letter-spacing: 0.05em !important;
    }

    .warning-box {
        background: rgba(255, 165, 0, 0.1);
        border: 1px solid rgba(255, 165, 0, 0.3);
        border-radius: 8px;
        padding: 0.8rem 1rem;
        color: #ffaa00;
        font-size: 0.85rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🏥 Clinical ICD-10 Coder</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">ICD-10 code prediction from clinical discharge summaries · PLM-ICD + Clinical-Longformer</div>', unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><div class="metric-value">0.5368</div><div class="metric-label">Best F1 Score</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><div class="metric-value">4096</div><div class="metric-label">Max Tokens</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><div class="metric-value">50</div><div class="metric-label">ICD Codes</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><div class="metric-value">108M</div><div class="metric-label">Parameters</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Sample texts ──────────────────────────────────────────────────────────────
SAMPLE_TEXTS = {
    "Heart Failure Patient": """Patient is a 72-year-old male presenting with progressive shortness of breath and bilateral lower extremity edema over the past 2 weeks. History of congestive heart failure, hypertension, and type 2 diabetes. BNP elevated at 1240 pg/mL. Echocardiogram shows LVEF of 35%. Creatinine 1.8 mg/dL. Started on IV furosemide with good response. Blood pressure controlled on lisinopril and metoprolol.""",
    "Pneumonia Patient": """65-year-old female admitted with fever, productive cough, and hypoxia. Chest X-ray shows right lower lobe consolidation consistent with pneumonia. White blood cell count 14,500. Started on IV ceftriaxone and azithromycin. History of COPD and hypertension. O2 saturation improved to 96% on 2L nasal cannula.""",
    "Kidney Disease Patient": """58-year-old male with end-stage renal disease on hemodialysis presenting with fluid overload and hyperkalemia. Potassium 6.2 mEq/L. BUN 85, creatinine 8.4. Underwent emergent dialysis with good response. History of type 2 diabetes and hypertension as primary causes of CKD.""",
}

# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📋 Discharge Summary Input")

    sample_choice = st.selectbox(
        "Load a sample note:",
        ["— paste your own —"] + list(SAMPLE_TEXTS.keys())
    )

    default_text = SAMPLE_TEXTS.get(sample_choice, "")
    text_input = st.text_area(
        "Discharge summary:",
        value=default_text,
        height=320,
        placeholder="Paste clinical discharge summary here...",
        label_visibility="collapsed"
    )

    col_thresh, col_topk = st.columns(2)
    with col_thresh:
        threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)
    with col_topk:
        top_k = st.slider("Max codes", 1, 20, 10)

    predict_btn = st.button("🔍 PREDICT ICD CODES")

    st.markdown('<div class="warning-box">⚠️ For clinical decision support only. Always verify with a certified medical coder.</div>', unsafe_allow_html=True)

with right_col:
    st.subheader("🎯 Predicted ICD-10 Codes")

    if predict_btn:
        if not text_input.strip():
            st.error("Please enter a discharge summary.")
        else:
            with st.spinner("Analyzing clinical note..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/predict",
                        json={
                            "text": text_input,
                            "top_k": top_k,
                            "threshold": threshold
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    result = response.json()
                    predictions = result['predictions']

                    if not predictions:
                        st.warning("No codes predicted above threshold. Try lowering the confidence threshold.")
                    else:
                        st.success(f"✅ {result['total_codes_predicted']} ICD codes predicted")

                        codes = [p['code'] for p in predictions]
                        confidences = [p['confidence'] * 100 for p in predictions]
                        descriptions = [p['description'] for p in predictions]

                        fig = go.Figure(go.Bar(
                            x=confidences,
                            y=codes,
                            orientation='h',
                            marker=dict(
                                color=confidences,
                                colorscale=[[0, '#004466'], [0.5, '#0099cc'], [1, '#00d4ff']],
                                showscale=False
                            ),
                            text=[f"{c:.1f}%" for c in confidences],
                            textposition='outside',
                            hovertemplate='<b>%{y}</b><br>%{customdata}<br>Confidence: %{x:.1f}%<extra></extra>',
                            customdata=descriptions
                        ))

                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family='IBM Plex Mono', color='#8892a4', size=11),
                            xaxis=dict(
                                title='Confidence (%)',
                                gridcolor='rgba(255,255,255,0.05)',
                                color='#8892a4',
                                range=[0, 115]
                            ),
                            yaxis=dict(
                                color='#00d4ff',
                                autorange='reversed'
                            ),
                            height=400,  # Fixed height to prevent layout shake
                            margin=dict(l=10, r=60, t=10, b=40)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown("**Detailed predictions:**")
                        df = pd.DataFrame([{
                            'Code': p['code'],
                            'Description': p['description'],
                            'Confidence': f"{p['confidence']*100:.1f}%"
                        } for p in predictions])
                        st.table(df)  # Static table — no flickering

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API. Run: `python app/main.py` in a separate terminal.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        st.info("Enter a discharge summary and click **PREDICT ICD CODES**.")

        st.markdown("**How it works:**")
        st.markdown("""
        1. Clinical-Longformer processes the full clinical note
        2. Label attention focuses each ICD code on relevant sections  
        3. 50 ICD codes scored with confidence levels
        4. Threshold filters low-confidence predictions
        """)

# ── Model comparison sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model Performance")
    perf_data = {
        'Model': ['TF-IDF + LR', 'ClinicalBERT', 'PLM-ICD'],
        'F1 Score': [0.5335, 0.4779, 0.5368]
    }
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)
    st.markdown("### Architecture")
    st.markdown("""
    - **Backbone**: Clinical-Longformer
    - **Mechanism**: Label Attention
    - **Parameters**: 108M
    - **Dataset**: MIMIC-IV
    - **Labels**: Top 50 ICD-10 codes
    """)
