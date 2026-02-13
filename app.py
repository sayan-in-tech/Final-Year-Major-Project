import os

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import pickle
import tensorflow as tf
import joblib
import whisper
import torch
import tempfile
import time
import cv2
from pathlib import Path
from audio_formatter import convert_to_wav
from torchvision import transforms
from models.cv.face_model import FaceEmotionCNN, EMOTIONS
from video_processor import process_video_emotions
import plotly.express as px
import plotly.graph_objects as go


try:
    torch_classes = getattr(torch, "classes", None)
    if torch_classes is not None:
        if not hasattr(torch_classes, "__path__"):
            class _TorchClassesPath(list):
                def __init__(self):
                    super().__init__()
                    self._path = []

                def __iter__(self):
                    return iter(self._path)

            torch_classes.__path__ = _TorchClassesPath()
        elif not hasattr(torch_classes.__path__, "_path"):
            torch_classes.__path__._path = []
    del torch_classes
except Exception:
    # Safe to continue even if patching torch classes fails.
    pass

BASE_DIR = Path(__file__).resolve().parent
TEXT_MODEL_PATH = BASE_DIR / "models" / "text" / "emotion_classifier_pipe_lr.pkl"

# ─── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Modern CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root variables ──────────────────────────────────── */
:root {
    --bg-primary:    #0a0a1a;
    --bg-secondary:  #111128;
    --bg-card:       rgba(17, 17, 40, 0.65);
    --glass-border:  rgba(255, 255, 255, 0.06);
    --glass-shadow:  0 8px 32px rgba(0, 0, 0, 0.35);
    --accent-1:      #7c3aed;   /* violet */
    --accent-2:      #06b6d4;   /* cyan */
    --accent-grad:   linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
    --text-primary:  #f0f0f8;
    --text-secondary:#a0a0c0;
    --text-muted:    #6b6b8d;
    --radius-sm:     10px;
    --radius-md:     16px;
    --radius-lg:     24px;
}

/* ── Global resets ───────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text-primary);
}

.main { padding: 0 !important; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1200px !important; }

/* ── Subtle noise texture ────────────────────────────── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(124, 58, 237, 0.12), transparent),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(6, 182, 212, 0.08), transparent);
    pointer-events: none;
    z-index: 0;
}

/* ── Typography ──────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}
p, li, span, label, div { color: var(--text-secondary); }
a { color: var(--accent-2); text-decoration: none; }

/* ── Glass card ──────────────────────────────────────── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px) saturate(1.4);
    -webkit-backdrop-filter: blur(20px) saturate(1.4);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    box-shadow: var(--glass-shadow);
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.45);
}

/* ── Hero header ─────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 3rem 2rem 2.5rem;
    margin-bottom: 2rem;
    background: var(--bg-card);
    backdrop-filter: blur(24px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    box-shadow: var(--glass-shadow);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.10) 0%, rgba(6, 182, 212, 0.08) 100%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: -0.5px;
    margin: 0;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-top: 0.75rem;
    font-weight: 400;
    line-height: 1.6;
}
.hero-chips {
    display: flex;
    justify-content: center;
    gap: 0.65rem;
    margin-top: 1.25rem;
    flex-wrap: wrap;
}
.hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    background: rgba(124, 58, 237, 0.12);
    border: 1px solid rgba(124, 58, 237, 0.25);
    color: #c4b5fd;
    letter-spacing: 0.3px;
}

/* ── Upload area ─────────────────────────────────────── */
.upload-wrap {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 2px dashed rgba(124, 58, 237, 0.25);
    border-radius: var(--radius-md);
    padding: 2.5rem 2rem;
    text-align: center;
    transition: border-color 0.3s, background 0.3s;
    margin-bottom: 1.5rem;
}
.upload-wrap:hover {
    border-color: rgba(124, 58, 237, 0.55);
    background: rgba(124, 58, 237, 0.04);
}
.upload-icon { font-size: 2.8rem; margin-bottom: 0.6rem; }
.upload-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}
.upload-desc {
    font-size: 0.92rem;
    color: var(--text-muted);
}

/* ── Section header ──────────────────────────────────── */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 2rem 0 1rem;
}
.section-hdr .icon-box {
    width: 40px; height: 40px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 10px;
    font-size: 1.2rem;
    flex-shrink: 0;
}
.section-hdr h3 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

/* ── Emotion badge ───────────────────────────────────── */
.emotion-badge {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 1.5rem 1rem;
    border-radius: var(--radius-md);
    text-align: center;
    min-height: 140px;
}
.emotion-badge .label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
    opacity: 0.7;
}
.emotion-badge .value {
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 1px;
}

/* ── Transcription box ───────────────────────────────── */
.transcript-box {
    background: rgba(124, 58, 237, 0.05);
    border-left: 3px solid var(--accent-1);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 1.25rem 1.5rem;
    font-size: 1rem;
    line-height: 1.8;
    color: var(--text-secondary);
    font-style: italic;
}

/* ── Final result hero ───────────────────────────────── */
.final-hero {
    position: relative;
    text-align: center;
    padding: 2.5rem 2rem;
    border-radius: var(--radius-lg);
    overflow: hidden;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
}
.final-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--accent-grad);
    opacity: 0.15;
    z-index: 0;
}
.final-hero > * { position: relative; z-index: 1; }
.final-hero .final-tag {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: var(--accent-2);
    margin-bottom: 0.4rem;
}
.final-hero .final-emotion {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: 2px;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0;
}
.final-hero .final-conf {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
}
.final-hero .modality-row {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 1.25rem;
    flex-wrap: wrap;
}
.final-hero .mod-chip {
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.3rem 0.9rem;
    border-radius: 9999px;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: var(--text-secondary);
}

/* ── Stat card ───────────────────────────────────────── */
.stat-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1rem;
    text-align: center;
}
.stat-card .stat-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text-primary);
}
.stat-card .stat-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-top: 0.3rem;
}

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}

/* ── Divider ─────────────────────────────────────────── */
.sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.25), transparent);
    margin: 2rem 0;
}

/* ── Streamlit overrides ─────────────────────────────── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] > div:first-child { padding: 0 !important; }
[data-testid="stFileUploader"] label { display: none !important; }
[data-testid="stMetric"] {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-md);
    padding: 1rem;
}
[data-testid="stMetricValue"] { color: var(--text-primary) !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

/* tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-card);
    border-radius: var(--radius-sm);
    padding: 0.25rem;
    border: 1px solid var(--glass-border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 0.5rem 1.25rem;
    font-weight: 600;
    color: var(--text-muted);
}
.stTabs [aria-selected="true"] {
    background: rgba(124, 58, 237, 0.15) !important;
    color: #c4b5fd !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* spinner */
.stSpinner > div { color: var(--accent-1) !important; }

/* video player */
video {
    border-radius: var(--radius-md);
    border: 1px solid var(--glass-border);
}

/* plotly background */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(124, 58, 237, 0.3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124, 58, 237, 0.5); }

/* animations */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.5s ease-out both; }
</style>
""", unsafe_allow_html=True)


# ─── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero fade-up">
    <div class="hero-title">Multimodal Emotion Recognition</div>
    <div class="hero-sub">
        Upload a video and let AI detect emotions from <b style="color:#c4b5fd">facial expressions</b>,
        <b style="color:#67e8f9">voice tone</b>, and <b style="color:#86efac">speech text</b> &mdash; all at once.
    </div>
    <div class="hero-chips">
        <span class="hero-chip">🎭 Face</span>
        <span class="hero-chip">🎙️ Voice</span>
        <span class="hero-chip">📝 Text</span>
        <span class="hero-chip">🧠 Fusion</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_utils():
    try:
        model = tf.keras.models.load_model("models/audio/emotion_recognition_model.keras")
        scaler_params = np.load("models/audio/scaler_params.npy", allow_pickle=True).item()
        mean = scaler_params['mean']
        scale = scaler_params['scale']
        with open("models/audio/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        return model, mean, scale, le
    except Exception as e:
        st.error(f"Error loading model or dependencies: {e}")
        return None, None, None, None


model, mean, scale, le = load_model_and_utils()


@st.cache_resource
def load_text_emotion_model():
    try:
        return joblib.load(TEXT_MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading text emotion model: {e}")
        return None


@st.cache_resource
def load_whisper_model():
    model_name = os.getenv("WHISPER_MODEL_NAME", "small.en")
    try:
        return whisper.load_model(model_name)
    except Exception as e:
        st.error(f"Error loading Whisper model '{model_name}': {e}")
        return None


pipe_lr = load_text_emotion_model()
whisper_model = load_whisper_model()


@st.cache_resource
def load_face_emotion_model():
    """Load face emotion detection model."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        face_cnn = FaceEmotionCNN(num_classes=7)
        model_path = BASE_DIR / "models" / "cv" / "face_model.pth"
        if model_path.exists():
            face_cnn.load_state_dict(torch.load(str(model_path), map_location=device))
        face_cnn.to(device)
        face_cnn.eval()
        return face_cnn, device
    except Exception as e:
        st.error(f"Error loading face emotion model: {e}")
        return None, None


face_model, face_device = load_face_emotion_model()

face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# ─── Mappings & colours ─────────────────────────────────────────────────────────
TEXT_TO_VOICE_EMOTION_MAP = {
    "anger": "angry",
    "joy": "happy",
    "fear": "fear",
    "disgust": "disgust",
    "sadness": "sad",
    "shame": "neutral",
    "surprise": "surprise",
}

emotion_colors = {
    "angry":    "#ef4444",
    "happy":    "#22c55e",
    "sad":      "#3b82f6",
    "fear":     "#a855f7",
    "disgust":  "#f97316",
    "surprise": "#06b6d4",
    "neutral":  "#94a3b8",
}

emotion_emoji = {
    "angry":    "😠",
    "happy":    "😄",
    "sad":      "😢",
    "fear":     "😨",
    "disgust":  "🤢",
    "surprise": "😲",
    "neutral":  "😐",
}


# ─── Plotly theme ────────────────────────────────────────────────────────────────
PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#a0a0c0", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
    margin=dict(l=40, r=20, t=30, b=40),
    bargap=0.35,
)


def make_emotion_bar_chart(labels, values, title_y="Probability", height=340):
    """Plotly bar chart with per-emotion colours and rounded bars."""
    colors = [emotion_colors.get(l.lower(), "#7c3aed") for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker=dict(color=colors, line=dict(width=0), cornerradius=6),
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, color="#a0a0c0"),
    ))
    fig.update_layout(**PLOTLY_DARK_LAYOUT, height=height,
                      yaxis_title=title_y, xaxis_title="Emotion")
    return fig


def make_emotion_timeline(times, emotions, confidences, height=340):
    """Plotly scatter-line coloured by emotion."""
    colors = [emotion_colors.get(e.lower(), "#7c3aed") for e in emotions]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=confidences, mode="lines+markers",
        marker=dict(color=colors, size=7, line=dict(width=1, color="rgba(255,255,255,0.15)")),
        line=dict(color="rgba(124,58,237,0.35)", width=2),
        text=emotions,
        hovertemplate="%{text}<br>Time: %{x:.1f}s<br>Conf: %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_DARK_LAYOUT, height=height,
                      xaxis_title="Time (s)", yaxis_title="Confidence")
    return fig


def make_emotion_count_chart(labels, counts, height=340):
    """Plotly bar chart for emotion occurrence counts."""
    colors = [emotion_colors.get(l.lower(), "#7c3aed") for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker=dict(color=colors, line=dict(width=0), cornerradius=6),
        text=counts, textposition="outside",
        textfont=dict(size=11, color="#a0a0c0"),
    ))
    fig.update_layout(**PLOTLY_DARK_LAYOUT, height=height,
                      yaxis_title="Count", xaxis_title="Emotion")
    return fig


def make_radar_chart(labels, values, height=380):
    """Plotly radar / polar chart for emotion probabilities."""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(124, 58, 237, 0.12)',
        line=dict(color='#7c3aed', width=2),
        marker=dict(size=5, color='#7c3aed'),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.06)",
                            tickfont=dict(size=9, color="#6b6b8d")),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                             tickfont=dict(size=11, color="#a0a0c0")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#a0a0c0"),
        margin=dict(l=60, r=60, t=40, b=40),
        height=height,
        showlegend=False,
    )
    return fig


# ─── Utility functions ──────────────────────────────────────────────────────────
def map_text_emotion_to_voice(emotion_label: str) -> str:
    return TEXT_TO_VOICE_EMOTION_MAP.get(emotion_label, emotion_label)


def aggregate_voice_probabilities(prob_matrix, classes):
    voice_probabilities = {}
    for cls, prob in zip(classes, prob_matrix[0]):
        voice_label = map_text_emotion_to_voice(cls)
        voice_probabilities[voice_label] = voice_probabilities.get(voice_label, 0.0) + float(prob)
    return voice_probabilities


def aggregate_multimodal_emotions(
    face_label=None, face_conf=0.0,
    audio_label=None, audio_conf=0.0,
    text_label=None, text_conf=0.0,
):
    """
    Aggregate emotion predictions from up to 3 modalities.
    Weights: face 35%, voice 55%, text 20%.
    Only active modalities (non-None label) participate.
    """
    w_face = 0.35
    w_audio = 0.55
    w_text = 0.20

    candidates = {}
    total_weight = 0.0

    if face_label is not None:
        candidates[face_label] = candidates.get(face_label, 0.0) + w_face * face_conf
        total_weight += w_face
    if audio_label is not None:
        candidates[audio_label] = candidates.get(audio_label, 0.0) + w_audio * audio_conf
        total_weight += w_audio
    if text_label is not None:
        candidates[text_label] = candidates.get(text_label, 0.0) + w_text * text_conf
        total_weight += w_text

    if not candidates or total_weight == 0:
        return None, 0.0

    final_label = max(candidates, key=candidates.get)
    final_conf = candidates[final_label] / total_weight
    return final_label, round(final_conf, 3)


# ─── Feature extraction & prediction ────────────────────────────────────────────
def extract_features(file_path, target_len=40):
    try:
        audio, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < target_len:
            pad_width = target_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :target_len]
        return mfcc
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None


def enhance_features(X):
    delta = np.diff(X, axis=1)
    delta = np.pad(delta, ((0, 0), (1, 0)), mode='edge')
    delta2 = np.diff(delta, axis=1)
    delta2 = np.pad(delta2, ((0, 0), (1, 0)), mode='edge')
    return np.stack([X, delta, delta2], axis=0)


def predict_audio_emotion(file):
    try:
        if model is None:
            return None, None, {}
        features = extract_features(file)
        if features is None:
            return None, None, {}

        enhanced = enhance_features(features)
        mfcc_std = (enhanced[0] - mean[:, np.newaxis]) / scale[:, np.newaxis]
        delta_std = (enhanced[1] - mean[:, np.newaxis]) / scale[:, np.newaxis]
        delta2_std = (enhanced[2] - mean[:, np.newaxis]) / scale[:, np.newaxis]

        enhanced_std = np.stack([mfcc_std, delta_std, delta2_std], axis=0)
        enhanced_std = enhanced_std[:, :, :40]
        input_data = np.expand_dims(enhanced_std[:, :, 0], axis=0)

        preds = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(preds)
        emotion = le.inverse_transform([predicted_index])[0]
        confidence = float(np.max(preds))

        all_emotions = le.classes_
        audio_probabilities = {}
        for idx, prob in enumerate(preds[0]):
            emotion_name = le.inverse_transform([idx])[0]
            audio_probabilities[emotion_name] = float(prob)

        return emotion, confidence, audio_probabilities
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, {}


def predict_audio_emotion_timeline(file_path: str, segment_duration: float = 3.0):
    """Split audio into segments and predict emotion for each."""
    if model is None:
        return []
    try:
        full_audio, sr = librosa.load(file_path, sr=22050)
    except Exception:
        return []

    timeline = []
    target_len = 40
    segment_samples = int(segment_duration * sr)

    offset = 0
    while offset < len(full_audio):
        chunk = full_audio[offset: offset + segment_samples]
        t = offset / sr

        if len(chunk) < sr * 0.5:
            break

        try:
            mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=40)
            if mfcc.shape[1] < target_len:
                pad_width = target_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :target_len]

            enhanced = enhance_features(mfcc)
            mfcc_std = (enhanced[0] - mean[:, np.newaxis]) / scale[:, np.newaxis]
            delta_std = (enhanced[1] - mean[:, np.newaxis]) / scale[:, np.newaxis]
            delta2_std = (enhanced[2] - mean[:, np.newaxis]) / scale[:, np.newaxis]
            enhanced_std = np.stack([mfcc_std, delta_std, delta2_std], axis=0)
            enhanced_std = enhanced_std[:, :, :40]
            input_data = np.expand_dims(enhanced_std[:, :, 0], axis=0)

            preds = model.predict(input_data, verbose=0)
            predicted_index = int(np.argmax(preds))
            emotion = le.inverse_transform([predicted_index])[0]
            confidence = float(np.max(preds))

            probs = {}
            for idx, prob in enumerate(preds[0]):
                probs[le.inverse_transform([idx])[0]] = float(prob)

            timeline.append({
                'time': round(t, 1),
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probs,
            })
        except Exception:
            pass

        offset += segment_samples

    return timeline


def transcribe_audio(file_path: str) -> str:
    if whisper_model is None:
        return ""
    try:
        result = whisper_model.transcribe(file_path, language="en")
        return result.get("text", "").strip()
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return ""


def predict_text_emotion(text: str):
    if not text or pipe_lr is None:
        return None, None, {}
    try:
        prediction = pipe_lr.predict([text])[0]
        mapped_prediction = map_text_emotion_to_voice(prediction)
        probability = pipe_lr.predict_proba([text])
        voice_probabilities = aggregate_voice_probabilities(probability, pipe_lr.classes_)
        mapped_confidence = voice_probabilities.get(mapped_prediction, 0.0)
        return mapped_prediction, mapped_confidence, voice_probabilities
    except Exception as e:
        st.error(f"Text emotion prediction error: {e}")
        return None, None, {}


def extract_audio_from_video(video_path: str) -> str:
    """Extract audio track from a video file and save as WAV."""
    from pydub import AudioSegment
    try:
        audio = AudioSegment.from_file(video_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            wav_path = tmp_wav.name
            audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        st.error(f"Error extracting audio from video: {e}")
        return None


# ─── Helper: section header HTML ────────────────────────────────────────────────
def section_header(icon: str, title: str, bg: str = "rgba(124,58,237,0.12)",
                   border: str = "rgba(124,58,237,0.25)"):
    return f"""
    <div class="section-hdr">
        <div class="icon-box" style="background:{bg}; border:1px solid {border};">{icon}</div>
        <h3>{title}</h3>
    </div>
    """


def emotion_badge_html(label_text: str, value_text: str,
                       bg: str = "rgba(124,58,237,0.08)",
                       border: str = "rgba(124,58,237,0.2)",
                       color: str = "#c4b5fd"):
    return f"""
    <div class="emotion-badge glass-card" style="background:{bg}; border-color:{border};">
        <div class="label" style="color:{color};">{label_text}</div>
        <div class="value" style="color:{color};">{value_text}</div>
    </div>
    """


# ─── Upload section ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-wrap fade-up">
    <div class="upload-icon">🎬</div>
    <div class="upload-title">Upload Your Video</div>
    <div class="upload-desc">Supported formats: MP4, AVI, MOV, MKV, WEBM</div>
</div>
""", unsafe_allow_html=True)

uploaded_video = st.file_uploader(
    "Upload video file",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    label_visibility="collapsed",
)

if uploaded_video is not None:
    # Save video to temporary file
    file_ext = os.path.splitext(uploaded_video.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_video:
        tmp_video.write(uploaded_video.read())
        video_path = tmp_video.name

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    st.video(video_path)

    # --- Placeholders for results from each modality ---
    face_label, face_confidence = None, 0.0
    audio_label, audio_confidence, audio_probabilities = None, None, {}
    audio_timeline = []
    text_label, text_confidence, text_probabilities = None, None, {}
    transcript = ""
    face_results = None

    # ═════════════════════════════════════════════════════════════
    # STEP 1: Extract audio
    # ═════════════════════════════════════════════════════════════
    with st.spinner("Extracting audio from video..."):
        wav_path = extract_audio_from_video(video_path)

    # ═════════════════════════════════════════════════════════════
    # STEP 2: Run all three analyses
    # ═════════════════════════════════════════════════════════════
    with st.spinner("Analyzing emotion from face, voice, and text..."):
        # --- Face ---
        if face_model is not None and face_device is not None:
            try:
                face_results = process_video_emotions(
                    video_path, face_model, face_transform, face_device, fps_sample=1,
                )
                if face_results['dominant_emotion'] is not None:
                    face_label = EMOTIONS[face_results['dominant_emotion']]
                    valid_confs = [c for c in face_results['confidences'] if c > 0]
                    face_confidence = float(np.mean(valid_confs)) if valid_confs else 0.0
            except Exception as e:
                st.warning(f"Face analysis error: {e}")

        # --- Voice ---
        if wav_path:
            audio_label, audio_confidence, audio_probabilities = predict_audio_emotion(wav_path)
            audio_timeline = predict_audio_emotion_timeline(wav_path, segment_duration=3.0)
            # --- Transcription + Text ---
            transcript = transcribe_audio(wav_path)
            text_label, text_confidence, text_probabilities = predict_text_emotion(transcript)

    # ═════════════════════════════════════════════════════════════
    # STEP 3: Aggregated result FIRST
    # ═════════════════════════════════════════════════════════════
    final_label, final_confidence = None, 0.0
    if face_label or audio_label or text_label:
        final_label, final_confidence = aggregate_multimodal_emotions(
            face_label=face_label, face_conf=face_confidence,
            audio_label=audio_label, audio_conf=audio_confidence if audio_confidence else 0.0,
            text_label=text_label, text_conf=text_confidence if text_confidence else 0.0,
        )

    if final_label:
        st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
        final_emoji = emotion_emoji.get(final_label.lower(), "🧠")
        final_color = emotion_colors.get(final_label.lower(), "#7c3aed")

        modalities_html = ""
        if face_label:
            modalities_html += f'<span class="mod-chip">🎭 Face: {face_label} ({face_confidence*100:.1f}%)</span>'
        if audio_label:
            modalities_html += f'<span class="mod-chip">🎙️ Voice: {audio_label} ({audio_confidence*100:.1f}%)</span>'
        if text_label:
            modalities_html += f'<span class="mod-chip">📝 Text: {text_label} ({text_confidence*100:.1f}%)</span>'

        st.markdown(f"""
        <div class="final-hero glass-card fade-up">
            <div class="final-tag">Aggregated Result</div>
            <div class="final-emotion">{final_emoji} {final_label.upper()}</div>
            <div class="final-conf">{final_confidence * 100:.1f}%</div>
            <div style="font-size:0.82rem; color:var(--text-muted); margin-top:0.5rem;">
                Weights &mdash; Face 35% &bull; Voice 55% &bull; Text 20%
            </div>
            <div class="modality-row">{modalities_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════
    # STEP 4: Three-column modality summary cards
    # ═════════════════════════════════════════════════════════════
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    st.markdown(section_header("🔬", "Individual Model Results"), unsafe_allow_html=True)

    col_face, col_voice, col_text = st.columns(3, gap="medium")

    # ── Face column ──────────────────────────────────────────────
    with col_face:
        if face_label:
            face_color = emotion_colors.get(face_label.lower(), "#7c3aed")
            face_emoji = emotion_emoji.get(face_label.lower(), "🎭")
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; border-top: 3px solid {face_color};">
                <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:1.5px; color:var(--text-muted); margin-bottom:0.6rem;">
                    🎭 Face
                </div>
                <div style="font-size:2.2rem; margin:0.3rem 0;">{face_emoji}</div>
                <div style="font-size:1.5rem; font-weight:900; color:{face_color};
                            letter-spacing:1px; margin:0.3rem 0;">
                    {face_label.upper()}
                </div>
                <div style="font-size:1.1rem; font-weight:700; color:#22c55e; margin-top:0.4rem;">
                    {face_confidence * 100:.1f}%
                </div>
                <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem;">
                    Avg confidence across frames
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; border-top: 3px solid #94a3b8; min-height:200px;
                        display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:1.5px; color:var(--text-muted); margin-bottom:0.6rem;">
                    🎭 Face
                </div>
                <div style="font-size:1.8rem; margin:0.5rem 0;">😶</div>
                <div style="font-size:0.85rem; color:var(--text-muted);">No faces detected</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Voice column ─────────────────────────────────────────────
    with col_voice:
        if audio_label:
            audio_color = emotion_colors.get(audio_label.lower(), "#7c3aed")
            audio_emoji = emotion_emoji.get(audio_label.lower(), "🎙️")
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; border-top: 3px solid {audio_color};">
                <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:1.5px; color:var(--text-muted); margin-bottom:0.6rem;">
                    🎙️ Voice
                </div>
                <div style="font-size:2.2rem; margin:0.3rem 0;">{audio_emoji}</div>
                <div style="font-size:1.5rem; font-weight:900; color:{audio_color};
                            letter-spacing:1px; margin:0.3rem 0;">
                    {audio_label.upper()}
                </div>
                <div style="font-size:1.1rem; font-weight:700; color:#22c55e; margin-top:0.4rem;">
                    {audio_confidence * 100:.1f}%
                </div>
                <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem;">
                    Overall voice confidence
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; border-top: 3px solid #94a3b8; min-height:200px;
                        display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:1.5px; color:var(--text-muted); margin-bottom:0.6rem;">
                    🎙️ Voice
                </div>
                <div style="font-size:1.8rem; margin:0.5rem 0;">🔇</div>
                <div style="font-size:0.85rem; color:var(--text-muted);">No voice emotion detected</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Text column ──────────────────────────────────────────────
    with col_text:
        if text_label:
            text_color = emotion_colors.get(text_label.lower(), "#7c3aed")
            text_emoji = emotion_emoji.get(text_label.lower(), "📝")
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; border-top: 3px solid {text_color};">
                <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:1.5px; color:var(--text-muted); margin-bottom:0.6rem;">
                    📝 Text
                </div>
                <div style="font-size:2.2rem; margin:0.3rem 0;">{text_emoji}</div>
                <div style="font-size:1.5rem; font-weight:900; color:{text_color};
                            letter-spacing:1px; margin:0.3rem 0;">
                    {text_label.upper()}
                </div>
                <div style="font-size:1.1rem; font-weight:700; color:#22c55e; margin-top:0.4rem;">
                    {text_confidence * 100:.1f}%
                </div>
                <div style="font-size:0.72rem; color:var(--text-muted); margin-top:0.3rem;">
                    Speech-to-text confidence
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; border-top: 3px solid #94a3b8; min-height:200px;
                        display:flex; flex-direction:column; align-items:center; justify-content:center;">
                <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase;
                            letter-spacing:1.5px; color:var(--text-muted); margin-bottom:0.6rem;">
                    📝 Text
                </div>
                <div style="font-size:1.8rem; margin:0.5rem 0;">💬</div>
                <div style="font-size:0.85rem; color:var(--text-muted);">No speech detected</div>
            </div>
            """, unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════
    # STEP 5: Transcription (if available)
    # ═════════════════════════════════════════════════════════════
    if transcript:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(section_header("📝", "Transcription"), unsafe_allow_html=True)
        st.markdown(f'<div class="transcript-box">{transcript}</div>',
                    unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════
    # STEP 6: Detailed breakdowns in tabs
    # ═════════════════════════════════════════════════════════════
    has_face_details = face_label and face_results and face_results.get('avg_probabilities')
    has_voice_details = audio_label and audio_probabilities
    has_text_details = text_label and text_probabilities

    if has_face_details or has_voice_details or has_text_details:
        st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
        st.markdown(section_header("📊", "Detailed Breakdowns"), unsafe_allow_html=True)

        detail_tabs = []
        if has_face_details:
            detail_tabs.append("🎭  Face Details")
        if has_voice_details:
            detail_tabs.append("🎙️  Voice Details")
        if has_text_details:
            detail_tabs.append("📝  Text Details")

        tabs = st.tabs(detail_tabs)
        tab_idx = 0

        # ── Face details ─────────────────────────────────────────
        if has_face_details:
            with tabs[tab_idx]:
                face_proba_labels = [EMOTIONS[i] for i in range(len(EMOTIONS))]
                face_proba_values = [face_results['avg_probabilities'][i] for i in range(len(EMOTIONS))]

                chart_col, radar_col = st.columns(2)
                with chart_col:
                    st.plotly_chart(make_emotion_bar_chart(face_proba_labels, face_proba_values),
                                   use_container_width=True)
                with radar_col:
                    st.plotly_chart(make_radar_chart(face_proba_labels, face_proba_values),
                                   use_container_width=True)

                # Face timeline
                if len(face_results['frame_times']) > 0 and any(
                        e is not None for e in face_results['emotions']):
                    st.markdown(section_header("📈", "Emotion Timeline"), unsafe_allow_html=True)
                    tl_times, tl_emotions, tl_confs = [], [], []
                    for i, (t, emotion_idx) in enumerate(
                            zip(face_results['frame_times'], face_results['emotions'])):
                        if emotion_idx is not None:
                            tl_times.append(t)
                            tl_emotions.append(EMOTIONS[emotion_idx])
                            tl_confs.append(face_results['confidences'][i])
                    if tl_times:
                        count_labels = [EMOTIONS[i] for i in face_results['emotion_counts'].keys()]
                        count_values = list(face_results['emotion_counts'].values())
                        tc1, tc2 = st.columns(2)
                        with tc1:
                            st.plotly_chart(make_emotion_timeline(tl_times, tl_emotions, tl_confs),
                                            use_container_width=True)
                        with tc2:
                            st.plotly_chart(make_emotion_count_chart(count_labels, count_values),
                                            use_container_width=True)
            tab_idx += 1

        # ── Voice details ────────────────────────────────────────
        if has_voice_details:
            with tabs[tab_idx]:
                ap_sorted = sorted(audio_probabilities.items(), key=lambda x: x[1], reverse=True)
                ap_labels, ap_values = zip(*ap_sorted)

                chart_col, radar_col = st.columns(2)
                with chart_col:
                    st.plotly_chart(make_emotion_bar_chart(list(ap_labels), list(ap_values)),
                                   use_container_width=True)
                with radar_col:
                    st.plotly_chart(make_radar_chart(list(ap_labels), list(ap_values)),
                                   use_container_width=True)

                # Voice timeline
                if audio_timeline and len(audio_timeline) > 1:
                    st.markdown(section_header("📈", "Voice Emotion Timeline"), unsafe_allow_html=True)
                    vtl_times = [e['time'] for e in audio_timeline]
                    vtl_emotions = [e['emotion'] for e in audio_timeline]
                    vtl_confs = [e['confidence'] for e in audio_timeline]

                    voice_emotion_counts = {}
                    for e in vtl_emotions:
                        voice_emotion_counts[e] = voice_emotion_counts.get(e, 0) + 1

                    vc1, vc2 = st.columns(2)
                    with vc1:
                        st.plotly_chart(make_emotion_timeline(vtl_times, vtl_emotions, vtl_confs),
                                        use_container_width=True)
                    with vc2:
                        st.plotly_chart(make_emotion_count_chart(
                            list(voice_emotion_counts.keys()),
                            list(voice_emotion_counts.values())),
                            use_container_width=True)
            tab_idx += 1

        # ── Text details ─────────────────────────────────────────
        if has_text_details:
            with tabs[tab_idx]:
                tp_sorted = sorted(text_probabilities.items(), key=lambda x: x[1], reverse=True)
                tp_labels, tp_values = zip(*tp_sorted)

                chart_col, radar_col = st.columns(2)
                with chart_col:
                    st.plotly_chart(make_emotion_bar_chart(list(tp_labels), list(tp_values)),
                                   use_container_width=True)
                with radar_col:
                    st.plotly_chart(make_radar_chart(list(tp_labels), list(tp_values)),
                                   use_container_width=True)

    # ═════════════════════════════════════════════════════════════
    # STEP 5: Video statistics
    # ═════════════════════════════════════════════════════════════
    if face_results:
        st.markdown(section_header("📊", "Video Statistics"), unsafe_allow_html=True)
        total_frames = len(face_results['emotions'])
        frames_with_faces = sum(1 for e in face_results['emotions'] if e is not None)
        duration = f"{face_results['frame_times'][-1]:.1f}s" if face_results['frame_times'] else "0s"
        detection_rate = f"{frames_with_faces / total_frames * 100:.1f}%" if total_frames > 0 else "0%"

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total_frames}</div>
                <div class="stat-label">Frames Analyzed</div>
            </div>""", unsafe_allow_html=True)
        with s2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{frames_with_faces}</div>
                <div class="stat-label">Faces Detected</div>
            </div>""", unsafe_allow_html=True)
        with s3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{duration}</div>
                <div class="stat-label">Duration</div>
            </div>""", unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{detection_rate}</div>
                <div class="stat-label">Detection Rate</div>
            </div>""", unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════
    # Cleanup temporary files
    # ═════════════════════════════════════════════════════════════
    for tmp in [video_path, wav_path]:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding: 1rem 0 2rem;">
    <span style="font-size:0.82rem; color:var(--text-muted); letter-spacing:0.5px;">
        Built with Streamlit &bull; TensorFlow &bull; PyTorch &bull; Whisper &bull; Plotly
    </span>
</div>
""", unsafe_allow_html=True)
