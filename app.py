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
# Set page config
st.set_page_config(page_title="Multimodal Emotion Recognition", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for dark theme styling
st.markdown("""
    <style>
    .main {
        padding: 2rem 1rem;
        background-color: #0E1117;
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .title-container {
        background: rgba(26, 26, 46, 0.8);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .emotion-card {
        background: rgba(26, 26, 46, 0.8);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.4);
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .emotion-label {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .confidence-label {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        color: #a0a0d0;
    }
    .upload-section {
        background: rgba(26, 26, 46, 0.8);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.4);
        margin: 2rem 0;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    .transcription-box {
        background: rgba(15, 52, 96, 0.6);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        font-size: 1.1rem;
        line-height: 1.6;
        color: #e0e0e0;
    }
    .final-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        text-align: center;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .final-emotion {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    .final-confidence {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    /* Dark theme for Streamlit components */
    .stMarkdown {
        color: #e0e0e0;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0 !important;
    }
    p {
        color: #b0b0b0 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 🎵 Title Section
st.markdown(
    """
    <div class="title-container">
        <h1 style='color: #a0a0ff; margin-bottom: 0.5rem;'>🧠 Multimodal Emotion Recognition</h1>
        <p style='font-size: 1.2rem; color: #b0b0d0; margin-top: 0.5rem;'>Upload a video to detect emotions from face, voice, and speech — all at once!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# 🔄 Load model and dependencies
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
        st.error(f"❌ Error loading model or dependencies: {e}")
        return None, None, None, None

model, mean, scale, le = load_model_and_utils()


@st.cache_resource
def load_text_emotion_model():
    try:
        return joblib.load(TEXT_MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Error loading text emotion model: {e}")
        return None


@st.cache_resource
def load_whisper_model():
    model_name = os.getenv("WHISPER_MODEL_NAME", "small.en")
    try:
        return whisper.load_model(model_name)
    except Exception as e:
        st.error(f"❌ Error loading Whisper model '{model_name}': {e}")
        return None


pipe_lr = load_text_emotion_model()
whisper_model = load_whisper_model()


@st.cache_resource
def load_face_emotion_model():
    """Load face emotion detection model."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FaceEmotionCNN(num_classes=7)
        model_path = BASE_DIR / "models" / "cv" / "face_model.pth"
        
        if model_path.exists():
            model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading face emotion model: {e}")
        return None, None


face_model, face_device = load_face_emotion_model()

# Face image transform
face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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
    "angry": "#FF4444",
    "happy": "#4CAF50",
    "sad": "#2196F3",
    "fear": "#9C27B0",
    "disgust": "#FF9800",
    "surprise": "#00BCD4",
    "neutral": "#9E9E9E",
}


PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e0e0e0",
    xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    margin=dict(l=40, r=20, t=30, b=40),
)


def make_emotion_bar_chart(labels, values, title_y="Probability", height=350):
    """Create a Plotly bar chart with per-emotion colours."""
    colors = [emotion_colors.get(l.lower(), "#667eea") for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(**PLOTLY_DARK_LAYOUT, height=height,
                      yaxis_title=title_y, xaxis_title="Emotion")
    return fig


def make_emotion_timeline(times, emotions, confidences, height=350):
    """Create a Plotly scatter-line chart coloured by emotion."""
    colors = [emotion_colors.get(e.lower(), "#667eea") for e in emotions]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=confidences, mode="lines+markers",
        marker=dict(color=colors, size=8),
        line=dict(color="rgba(102,126,234,0.4)", width=1),
        text=emotions, hovertemplate="%{text}<br>Time: %{x:.1f}s<br>Conf: %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_DARK_LAYOUT, height=height,
                      xaxis_title="Time (seconds)", yaxis_title="Confidence")
    return fig


def make_emotion_count_chart(labels, counts, height=350):
    """Create a Plotly bar chart for emotion occurrence counts."""
    colors = [emotion_colors.get(l.lower(), "#667eea") for l in labels]
    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color=colors,
        text=counts, textposition="outside",
    ))
    fig.update_layout(**PLOTLY_DARK_LAYOUT, height=height,
                      yaxis_title="Count", xaxis_title="Emotion")
    return fig


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
    Weights: face 40%, voice 40%, text 20%.
    Only active modalities (non-None label) participate.
    """
    w_face = 0.40
    w_audio = 0.40
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

# 🎯 Feature Extraction
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
        st.error(f"❌ Error extracting features: {e}")
        return None

# 🔁 Delta & Delta²
def enhance_features(X):
    delta = np.diff(X, axis=1)
    delta = np.pad(delta, ((0, 0), (1, 0)), mode='edge')
    delta2 = np.diff(delta, axis=1)
    delta2 = np.pad(delta2, ((0, 0), (1, 0)), mode='edge')
    return np.stack([X, delta, delta2], axis=0)

# 🔍 Prediction
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
        
        # Create probability dictionary for all emotions
        all_emotions = le.classes_
        audio_probabilities = {}
        for idx, prob in enumerate(preds[0]):
            emotion_name = le.inverse_transform([idx])[0]
            audio_probabilities[emotion_name] = float(prob)
        
        return emotion, confidence, audio_probabilities
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None, {}


def predict_audio_emotion_timeline(file_path: str, segment_duration: float = 3.0):
    """
    Split audio into segments and predict emotion for each, returning a timeline.

    Args:
        file_path: Path to WAV file
        segment_duration: Duration of each segment in seconds

    Returns:
        List of dicts with keys: time, emotion, confidence, probabilities
    """
    if model is None:
        return []

    try:
        full_audio, sr = librosa.load(file_path, sr=22050)
    except Exception:
        return []

    total_duration = len(full_audio) / sr
    timeline = []
    target_len = 40
    segment_samples = int(segment_duration * sr)

    offset = 0
    while offset < len(full_audio):
        chunk = full_audio[offset : offset + segment_samples]
        t = offset / sr

        # Skip very short trailing chunks
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
        st.error(f"❌ Transcription error: {e}")
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
        st.error(f"❌ Text emotion prediction error: {e}")
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
        st.error(f"❌ Error extracting audio from video: {e}")
        return None


# 📤 Video Upload Section
st.markdown("""
    <div class="upload-section">
        <h2 style='text-align: center; color: #a0a0ff; margin-bottom: 1rem;'>🎥 Upload Your Video File</h2>
        <p style='text-align: center; color: #b0b0d0;'>We'll analyze emotion from <b>face</b>, <b>voice</b>, and <b>speech text</b> — all from one video.</p>
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

    # Display video
    st.markdown("<br>", unsafe_allow_html=True)
    st.video(video_path)

    # --- Placeholders for results from each modality ---
    face_label, face_confidence = None, 0.0
    audio_label, audio_confidence, audio_probabilities = None, None, {}
    audio_timeline = []
    text_label, text_confidence, text_probabilities = None, None, {}
    transcript = ""
    face_results = None

    # =========================================================
    # STEP 1: Extract audio from video
    # =========================================================
    with st.spinner("🎵 Extracting audio from video..."):
        wav_path = extract_audio_from_video(video_path)

    # =========================================================
    # STEP 2: Run all three analyses
    # =========================================================
    with st.spinner("🔍 Analyzing emotion from face, voice, and text..."):
        # --- Face emotion analysis ---
        if face_model is not None and face_device is not None:
            try:
                face_results = process_video_emotions(
                    video_path,
                    face_model,
                    face_transform,
                    face_device,
                    fps_sample=1,
                )
                if face_results['dominant_emotion'] is not None:
                    face_label = EMOTIONS[face_results['dominant_emotion']]
                    valid_confs = [c for c in face_results['confidences'] if c > 0]
                    face_confidence = float(np.mean(valid_confs)) if valid_confs else 0.0
            except Exception as e:
                st.warning(f"⚠️ Face analysis error: {e}")

        # --- Voice emotion analysis ---
        if wav_path:
            audio_label, audio_confidence, audio_probabilities = predict_audio_emotion(wav_path)
            audio_timeline = predict_audio_emotion_timeline(wav_path, segment_duration=3.0)

            # --- Transcription + Text emotion analysis ---
            transcript = transcribe_audio(wav_path)
            text_label, text_confidence, text_probabilities = predict_text_emotion(transcript)

    # =========================================================
    # STEP 3: Display individual results
    # =========================================================

    # --- 1. Face Emotion ---
    if face_label:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="emotion-card">
                <h2 class="section-header" style='color: #667eea;'>😊 Face Emotion Analysis</h2>
            </div>
        """, unsafe_allow_html=True)

        face_cols = st.columns([1, 1])
        face_color = emotion_colors.get(face_label.lower(), "#667eea")

        with face_cols[0]:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, {face_color}33 0%, {face_color}55 100%); border-radius: 10px; border: 1px solid {face_color}66;'>
                    <h3 style='color: #d0d0d0; margin-bottom: 0.5rem;'>Dominant Emotion</h3>
                    <h2 class="emotion-label" style='color: {face_color};'>{face_label.upper()}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with face_cols[1]:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #4CAF5033 0%, #4CAF5055 100%); border-radius: 10px; border: 1px solid #4CAF5066;'>
                    <h3 style='color: #d0d0d0; margin-bottom: 0.5rem;'>Average Confidence</h3>
                    <h2 class="confidence-label" style='color: #4CAF50;'>{face_confidence * 100:.2f}%</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Face probability chart
        if face_results and face_results['avg_probabilities']:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Face Emotion Probabilities")
            face_proba_labels = [EMOTIONS[i] for i in range(len(EMOTIONS))]
            face_proba_values = [face_results['avg_probabilities'][i] for i in range(len(EMOTIONS))]
            st.plotly_chart(make_emotion_bar_chart(face_proba_labels, face_proba_values),
                            use_container_width=True)

        # Face emotion timeline
        if face_results and len(face_results['frame_times']) > 0 and any(e is not None for e in face_results['emotions']):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Face Emotion Timeline")
            tl_times, tl_emotions, tl_confs = [], [], []
            for i, (t, emotion_idx) in enumerate(zip(face_results['frame_times'], face_results['emotions'])):
                if emotion_idx is not None:
                    tl_times.append(t)
                    tl_emotions.append(EMOTIONS[emotion_idx])
                    tl_confs.append(face_results['confidences'][i])
            if tl_times:
                count_labels = [EMOTIONS[i] for i in face_results['emotion_counts'].keys()]
                count_values = list(face_results['emotion_counts'].values())
                tcol1, tcol2 = st.columns(2)
                with tcol1:
                    st.plotly_chart(make_emotion_timeline(tl_times, tl_emotions, tl_confs),
                                    use_container_width=True)
                with tcol2:
                    st.plotly_chart(make_emotion_count_chart(count_labels, count_values),
                                    use_container_width=True)
    else:
        st.warning("⚠️ No faces detected in the video.")

    # --- 2. Voice Emotion ---
    if audio_label:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="emotion-card">
                <h2 class="section-header" style='color: #667eea;'>🔊 Voice Emotion Analysis</h2>
            </div>
        """, unsafe_allow_html=True)

        audio_cols = st.columns([1, 1])
        audio_color = emotion_colors.get(audio_label.lower(), "#667eea")

        with audio_cols[0]:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, {audio_color}33 0%, {audio_color}55 100%); border-radius: 10px; border: 1px solid {audio_color}66;'>
                    <h3 style='color: #d0d0d0; margin-bottom: 0.5rem;'>Emotion</h3>
                    <h2 class="emotion-label" style='color: {audio_color};'>{audio_label.upper()}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with audio_cols[1]:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #4CAF5033 0%, #4CAF5055 100%); border-radius: 10px; border: 1px solid #4CAF5066;'>
                    <h3 style='color: #d0d0d0; margin-bottom: 0.5rem;'>Confidence</h3>
                    <h2 class="confidence-label" style='color: #4CAF50;'>{audio_confidence * 100:.2f}%</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if audio_probabilities:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Voice Emotion Probabilities")
            ap_labels = list(audio_probabilities.keys())
            ap_values = list(audio_probabilities.values())
            # Sort descending
            ap_sorted = sorted(zip(ap_labels, ap_values), key=lambda x: x[1], reverse=True)
            ap_labels, ap_values = zip(*ap_sorted)
            st.plotly_chart(make_emotion_bar_chart(list(ap_labels), list(ap_values)),
                            use_container_width=True)

        # Voice emotion timeline
        if audio_timeline and len(audio_timeline) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Voice Emotion Timeline")

            vtl_times = [e['time'] for e in audio_timeline]
            vtl_emotions = [e['emotion'] for e in audio_timeline]
            vtl_confs = [e['confidence'] for e in audio_timeline]

            voice_emotion_counts = {}
            for e in vtl_emotions:
                voice_emotion_counts[e] = voice_emotion_counts.get(e, 0) + 1

            vcol1, vcol2 = st.columns(2)
            with vcol1:
                st.plotly_chart(make_emotion_timeline(vtl_times, vtl_emotions, vtl_confs),
                                use_container_width=True)
            with vcol2:
                st.plotly_chart(make_emotion_count_chart(
                    list(voice_emotion_counts.keys()),
                    list(voice_emotion_counts.values())),
                    use_container_width=True)
    elif wav_path:
        st.warning("⚠️ Unable to detect voice emotion from video audio.")

    # --- 3. Transcription ---
    if transcript:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="emotion-card">
                <h2 class="section-header" style='color: #667eea;'>📝 Transcription</h2>
                <div class="transcription-box">
                    {transcript}
                </div>
            </div>
        """.format(transcript=transcript), unsafe_allow_html=True)

    # --- 4. Text Emotion ---
    if text_label:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="emotion-card">
                <h2 class="section-header" style='color: #667eea;'>📚 Text Emotion Analysis</h2>
            </div>
        """, unsafe_allow_html=True)

        text_cols = st.columns([1, 1])
        text_emotion_color = emotion_colors.get(text_label.lower(), "#667eea")

        with text_cols[0]:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, {text_emotion_color}33 0%, {text_emotion_color}55 100%); border-radius: 10px; border: 1px solid {text_emotion_color}66;'>
                    <h3 style='color: #d0d0d0; margin-bottom: 0.5rem;'>Emotion</h3>
                    <h2 class="emotion-label" style='color: {text_emotion_color};'>{text_label.upper()}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with text_cols[1]:
            st.markdown(
                f"""
                <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #4CAF5033 0%, #4CAF5055 100%); border-radius: 10px; border: 1px solid #4CAF5066;'>
                    <h3 style='color: #d0d0d0; margin-bottom: 0.5rem;'>Confidence</h3>
                    <h2 class="confidence-label" style='color: #4CAF50;'>{text_confidence * 100:.2f}%</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if text_probabilities:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📊 Text Emotion Probabilities")
            tp_labels = list(text_probabilities.keys())
            tp_values = list(text_probabilities.values())
            tp_sorted = sorted(zip(tp_labels, tp_values), key=lambda x: x[1], reverse=True)
            tp_labels, tp_values = zip(*tp_sorted)
            st.plotly_chart(make_emotion_bar_chart(list(tp_labels), list(tp_values)),
                            use_container_width=True)

    # =========================================================
    # STEP 4: Final Aggregated Emotion (3-way)
    # =========================================================
    if face_label or audio_label or text_label:
        final_label, final_confidence = aggregate_multimodal_emotions(
            face_label=face_label, face_conf=face_confidence,
            audio_label=audio_label, audio_conf=audio_confidence if audio_confidence else 0.0,
            text_label=text_label, text_conf=text_confidence if text_confidence else 0.0,
        )

        if final_label:
            st.markdown("<br>", unsafe_allow_html=True)
            final_emotion_color = emotion_colors.get(final_label.lower(), "#667eea")

            # Build modality breakdown string
            modalities_used = []
            if face_label:
                modalities_used.append(f"Face: {face_label} ({face_confidence*100:.1f}%)")
            if audio_label:
                modalities_used.append(f"Voice: {audio_label} ({audio_confidence*100:.1f}%)")
            if text_label:
                modalities_used.append(f"Text: {text_label} ({text_confidence*100:.1f}%)")
            modality_str = " &nbsp;|&nbsp; ".join(modalities_used)

            st.markdown(f"""
                <div class="final-result">
                    <h2 style='margin-bottom: 0.5rem; font-size: 1.6rem;'>🧠 Final Aggregated Emotion</h2>
                    <p style='color: rgba(255,255,255,0.7); font-size: 0.95rem; margin-bottom: 1rem;'>Weights: Face 40% &nbsp;|&nbsp; Voice 40% &nbsp;|&nbsp; Text 20%</p>
                    <div class="final-emotion" style='color: white;'>{final_label.upper()}</div>
                    <div class="final-confidence" style='color: rgba(255, 255, 255, 0.9);'>{final_confidence * 100:.2f}%</div>
                    <p style='color: rgba(255,255,255,0.6); font-size: 0.9rem; margin-top: 1rem;'>{modality_str}</p>
                </div>
            """, unsafe_allow_html=True)

    # =========================================================
    # STEP 5: Video Statistics
    # =========================================================
    if face_results:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Video Analysis Statistics")
        stat_cols = st.columns(4)
        total_frames = len(face_results['emotions'])
        frames_with_faces = sum(1 for e in face_results['emotions'] if e is not None)

        with stat_cols[0]:
            st.metric("Frames Analyzed", total_frames)
        with stat_cols[1]:
            st.metric("Frames with Faces", frames_with_faces)
        with stat_cols[2]:
            st.metric("Video Duration", f"{face_results['frame_times'][-1]:.1f}s" if face_results['frame_times'] else "0s")
        with stat_cols[3]:
            st.metric("Face Detection Rate", f"{frames_with_faces/total_frames*100:.1f}%" if total_frames > 0 else "0%")

    # =========================================================
    # Cleanup temporary files
    # =========================================================
    for tmp in [video_path, wav_path]:
        if tmp:
            try:
                os.remove(tmp)
            except Exception:
                pass
