# ============================================
#   Parkinson's Disease Detection App
# ============================================

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="Parkinson's Disease Detector",
    page_icon="🧠",
    layout="centered"
)

# ── Load Model & Scaler ───────────────────────
@st.cache_resource
def load_model_scaler():
    model = load_model('parkinsons_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_scaler()

# ── Header ────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#4FC3F7;'>
    🧠 Parkinson's Disease Detection
    </h1>
    <p style='text-align:center; color:gray;'>
    Using Artificial Neural Network (ANN) | Voice Biomarker Analysis
    </p>
    <hr>
""", unsafe_allow_html=True)

# ── Sidebar Info ──────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Parkinson%27s_disease_JPEG.jpg/220px-Parkinson%27s_disease_JPEG.jpg")
    st.markdown("### ℹ️ About")
    st.info("""
    This app uses a trained ANN model
    to detect Parkinson's Disease from
    voice measurements.
    
    **Dataset:** UCI Parkinson's Dataset
    **Model:** Multi-Layer ANN
    **Accuracy:** ~94-97%
    """)
    st.markdown("### 👥 Group Members")
    st.success("Enter your group members here")

# ── Input Features ────────────────────────────
st.markdown("### 🎙️ Enter Voice Feature Values")
st.markdown("*Adjust the sliders below to input patient voice measurements*")

col1, col2, col3 = st.columns(3)

with col1:
    MDVP_Fo    = st.slider('MDVP:Fo(Hz)',    80.0,  270.0, 150.0)
    MDVP_Fhi   = st.slider('MDVP:Fhi(Hz)',   100.0, 600.0, 200.0)
    MDVP_Flo   = st.slider('MDVP:Flo(Hz)',   60.0,  240.0, 110.0)
    MDVP_Jitter= st.slider('MDVP:Jitter(%)', 0.001, 0.03,  0.005)
    MDVP_Jitter_Abs = st.slider('MDVP:Jitter(Abs)', 0.0,  0.0003, 0.00003)
    MDVP_RAP   = st.slider('MDVP:RAP',       0.0,   0.02,  0.003)
    MDVP_PPQ   = st.slider('MDVP:PPQ',       0.0,   0.02,  0.003)

with col2:
    Jitter_DDP = st.slider('Jitter:DDP',     0.0,   0.07,  0.009)
    MDVP_Shimmer = st.slider('MDVP:Shimmer', 0.01,  0.12,  0.03)
    MDVP_Shimmer_dB = st.slider('MDVP:Shimmer(dB)', 0.1, 1.5, 0.3)
    Shimmer_APQ3 = st.slider('Shimmer:APQ3', 0.0,  0.06,  0.015)
    Shimmer_APQ5 = st.slider('Shimmer:APQ5', 0.0,  0.08,  0.02)
    MDVP_APQ   = st.slider('MDVP:APQ',       0.0,   0.14,  0.025)

with col3:
    Shimmer_DDA = st.slider('Shimmer:DDA',   0.0,   0.17,  0.045)
    NHR         = st.slider('NHR',           0.0,   0.32,  0.025)
    HNR         = st.slider('HNR',           8.0,   35.0,  22.0)
    RPDE        = st.slider('RPDE',          0.25,  0.69,  0.5)
    DFA         = st.slider('DFA',           0.57,  0.83,  0.72)
    spread1     = st.slider('Spread1',      -7.96, -2.43, -5.68)
    spread2     = st.slider('Spread2',       0.006, 0.45,  0.22)
    D2          = st.slider('D2',            1.42,  3.67,  2.38)
    PPE         = st.slider('PPE',           0.04,  0.53,  0.21)

# ── Predict Button ────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)

if st.button("🔍 Predict Now", use_container_width=True):

    features = np.array([[
        MDVP_Fo, MDVP_Fhi, MDVP_Flo,
        MDVP_Jitter, MDVP_Jitter_Abs,
        MDVP_RAP, MDVP_PPQ, Jitter_DDP,
        MDVP_Shimmer, MDVP_Shimmer_dB,
        Shimmer_APQ3, Shimmer_APQ5,
        MDVP_APQ, Shimmer_DDA,
        NHR, HNR, RPDE, DFA,
        spread1, spread2, D2, PPE,
        0.0  # placeholder
    ]])

    features_scaled = scaler.transform(features[:, :22])
    prediction = model.predict(features_scaled)
    probability = float(prediction[0][0])

    st.markdown("### 🩺 Prediction Result")

    if probability > 0.5:
        st.markdown(f"""
        <div style='background:#ff4b4b22; padding:20px;
        border-radius:10px; border:2px solid #ff4b4b;
        text-align:center;'>
        <h2 style='color:#ff4b4b;'>
        🔴 HIGH RISK — Parkinson's Detected</h2>
        <h3>Confidence: {probability*100:.1f}%</h3>
        <p>Please consult a neurologist immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#00c85322; padding:20px;
        border-radius:10px; border:2px solid #00c853;
        text-align:center;'>
        <h2 style='color:#00c853;'>
        🟢 LOW RISK — No Parkinson's Detected</h2>
        <h3>Confidence: {(1-probability)*100:.1f}%</h3>
        <p>Voice patterns appear normal.</p>
        </div>
        """, unsafe_allow_html=True)

    # Progress bar
    st.markdown("#### Risk Probability:")
    st.progress(probability)

# ── Footer ────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:gray; font-size:12px;'>
ANN Project | Artificial Neural Networks Lab |
Parkinson's Disease Early Detection System
</p>
""", unsafe_allow_html=True)