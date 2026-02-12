import streamlit as st
import torch
import numpy as np
from PIL import Image
import librosa
from torchvision import transforms
from models.face_model import FaceEmotionCNN, EMOTIONS
from models.voice_model import VoiceStressCNN, extract_mfcc, STRESS_LEVELS
from models.text_model import TextSentimentAnalyzer, SENTIMENTS
import tempfile
import pandas as pd
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Page config MUST BE FIRST - before any other Streamlit command!
st.set_page_config(page_title="Emotion Detector", layout="wide", page_icon="🧠")

# Custom CSS for beautiful background (AFTER set_page_config)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Make text white/visible */
    .stApp h1, .stApp h2, .stApp h3, .stApp p {
        color: white !important;
    }
    
    /* Style cards with glass effect */
    .stApp [data-testid="stVerticalBlock"] > div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: white !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton > button {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 8px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.3);
        border-color: rgba(255, 255, 255, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)

# Faster model loading with better caching
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    device = torch.device('cpu')  # Use CPU for faster startup
    
    # Face model with trained weights
    face_model = FaceEmotionCNN(num_classes=7)
    if os.path.exists('checkpoints/face_model.pth'):
        face_model.load_state_dict(torch.load('checkpoints/face_model.pth', map_location=device))
        print("✅ Loaded trained face model!")
    face_model.to(device)
    face_model.eval()
    
    # Voice model
    voice_model = VoiceStressCNN(num_classes=5)
    voice_model.to(device)
    voice_model.eval()
    
    # Text model
    text_model = TextSentimentAnalyzer()
    
    return face_model, voice_model, text_model, device

# Load models
with st.spinner("🚀 Loading AI models..."):
    face_model, voice_model, text_model, device = load_models()

# Transform
face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TRIGGER_WORDS = ['stress', 'anxious', 'worried', 'depressed', 'tired', 'overwhelmed']

if 'history' not in st.session_state:
    st.session_state.history = []

# Title
st.title("🧠 Multimodal Emotion & Stress Detection")
st.markdown("### Analyze emotions from face, voice, and text")

# Show if trained model is loaded
if os.path.exists('checkpoints/face_model.pth'):
    st.success("✅ Using trained face emotion model (64% accuracy)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Mode")
    mode = st.radio("", ["📊 Analyze", "📈 History"], label_visibility="collapsed")

if mode == "📊 Analyze":
    col1, col2, col3 = st.columns(3)
    
    face_idx = None
    voice_idx = None
    text_idx = None
    
    # FACE
    with col1:
        st.subheader("📸 Face Emotion")
        img_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'], key="face")
        
        if img_file:
            image = Image.open(img_file).convert('RGB')
            st.image(image, use_column_width=True)
            
            with st.spinner("Analyzing face..."):
                img_tensor = face_transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = face_model(img_tensor)
                    probs = torch.nn.functional.softmax(output, dim=1)
                    face_idx = torch.argmax(probs, dim=1).item()
                    confidence = probs[0][face_idx].item()
                
                st.success(f"**{EMOTIONS[face_idx].upper()}**")
                st.metric("Confidence", f"{confidence*100:.1f}%")
                
                # Show probability chart
                emotion_df = pd.DataFrame({
                    'Emotion': EMOTIONS,
                    'Probability': probs[0].cpu().numpy()
                })
                st.bar_chart(emotion_df.set_index('Emotion'))
    
    # VOICE
    with col2:
        st.subheader("🎤 Voice Stress")
        audio_file = st.file_uploader("Upload audio", type=['wav', 'mp3'], key="voice")
        
        if audio_file:
            st.audio(audio_file)
            
            with st.spinner("Analyzing voice..."):
                try:
                    # Save audio temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(audio_file.read())
                        tmp_path = tmp.name
                    
                    # Use feature-based analysis (NO TRAINING NEEDED!)
                    from models.voice_model import analyze_voice_stress
                    voice_idx, probs, features = analyze_voice_stress(tmp_path)
                    
                    # Display results
                    st.success(f"**{STRESS_LEVELS[voice_idx].upper()}**")
                    st.metric("Confidence", f"{probs[voice_idx]*100:.1f}%")
                    
                    # Show audio features
                    with st.expander("📊 Audio Features"):
                        st.write(f"🔊 Energy: {features.get('energy', 0):.4f}")
                        st.write(f"🎵 Average Pitch: {features.get('avg_pitch', 0):.1f} Hz")
                        st.write(f"📈 Pitch Variation: {features.get('pitch_variation', 0):.1f}")
                        st.write(f"⚡ Speaking Rate: {features.get('speaking_rate', 0):.4f}")
                    
                    # Show probability chart
                    stress_df = pd.DataFrame({
                        'Level': STRESS_LEVELS,
                        'Probability': probs
                    })
                    st.bar_chart(stress_df.set_index('Level'))
                    
                    os.remove(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error analyzing audio: {str(e)}")
                    voice_idx = None
    
    # TEXT
    with col3:
        st.subheader("📝 Text Sentiment")
        text_input = st.text_area("Type your text here", height=100, key="text")
        
        if st.button("🔍 Analyze Text", type="primary") and text_input:
            with st.spinner("Analyzing text..."):
                probs = text_model.analyze(text_input)
                text_idx = np.argmax(probs)
                confidence = probs[text_idx]
                
                st.success(f"**{SENTIMENTS[text_idx].upper()}**")
                st.metric("Confidence", f"{confidence*100:.1f}%")
                
                triggers = [w for w in TRIGGER_WORDS if w in text_input.lower()]
                if triggers:
                    st.warning(f"⚠️ Stress keywords: {', '.join(triggers)}")
                
                sent_df = pd.DataFrame({
                    'Sentiment': SENTIMENTS,
                    'Probability': probs
                })
                st.bar_chart(sent_df.set_index('Sentiment'))
    
    # Combined Score
    st.markdown("---")
    st.subheader("🎯 Combined Stress Score")
    
    if face_idx is not None or voice_idx is not None or text_idx is not None:
        face_stress = face_idx / 6.0 if face_idx is not None else 0.5
        voice_stress = voice_idx / 4.0 if voice_idx is not None else 0.5
        text_stress = (2 - text_idx) / 2.0 if text_idx is not None else 0.5
        
        combined = (face_stress * 0.4 + voice_stress * 0.4 + text_stress * 0.2) * 100
        
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.metric("Overall Stress Level", f"{combined:.1f}%")
            st.progress(int(combined) / 100)
        
        with col_b:
            if combined < 30:
                st.success("😊 Low Stress")
            elif combined < 60:
                st.info("😐 Moderate Stress")
            else:
                st.error("😰 High Stress")
        
        if st.button("💾 Save to History"):
            entry = {
                'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'face': EMOTIONS[face_idx] if face_idx is not None else 'N/A',
                'voice': STRESS_LEVELS[voice_idx] if voice_idx is not None else 'N/A',
                'text': SENTIMENTS[text_idx] if text_idx is not None else 'N/A',
                'score': round(combined, 1)
            }
            st.session_state.history.append(entry)
            st.success("✅ Saved to history!")
    else:
        st.info("👆 Upload at least one input (face, voice, or text) to see results")

else:  # History
    st.subheader("📈 Your Stress History")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Line chart
        st.line_chart(df.set_index('time')['score'])
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Analyses", len(df))
        with col2:
            st.metric("📈 Average Stress", f"{df['score'].mean():.1f}%")
        with col3:
            st.metric("🔴 Highest", f"{df['score'].max():.1f}%")
        with col4:
            st.metric("🟢 Lowest", f"{df['score'].min():.1f}%")
        
        # Full table
        st.dataframe(df, use_container_width=True)
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download History (CSV)",
            data=csv,
            file_name=f"stress_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("📭 No history yet! Start analyzing to track your stress over time.")

st.markdown("---")
st.caption("Built with ❤️ using PyTorch & Streamlit | 3-Day ML Project")