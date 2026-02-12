import torch
import torch.nn as nn
import librosa
import numpy as np

class VoiceStressCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(VoiceStressCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def analyze_voice_stress(audio_path):
    """
    Analyze voice stress using audio features (NO TRAINING NEEDED)
    Returns: stress_level (0-4) where 0=calm, 4=extreme
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, duration=3, sr=22050)
        
        # Feature 1: Energy/Volume (louder = more stress)
        rms = librosa.feature.rms(y=y)[0]
        energy = np.mean(rms)
        energy_std = np.std(rms)
        
        # Feature 2: Pitch (higher pitch = more stress)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_variation = np.std(pitch_values) if len(pitch_values) > 0 else 0
        
        # Feature 3: Speaking rate (faster = more stress)
        zero_crossings = librosa.zero_crossings(y, pad=False)
        zcr = sum(zero_crossings) / len(y)
        
        # Feature 4: Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # Calculate stress score (0-10 scale)
        stress_score = 0
        
        # Energy contribution (0-3 points)
        if energy > 0.02:
            stress_score += 0.5
        if energy > 0.05:
            stress_score += 1
        if energy > 0.1:
            stress_score += 1
        if energy > 0.15:
            stress_score += 0.5
            
        # Pitch contribution (0-3 points)
        if avg_pitch > 150:
            stress_score += 0.5
        if avg_pitch > 200:
            stress_score += 1
        if pitch_variation > 30:
            stress_score += 1
        if pitch_variation > 60:
            stress_score += 0.5
            
        # Speaking rate contribution (0-2 points)
        if zcr > 0.05:
            stress_score += 1
        if zcr > 0.1:
            stress_score += 1
            
        # Spectral contribution (0-2 points)
        if spectral_centroid > 1500:
            stress_score += 1
        if spectral_centroid > 2500:
            stress_score += 1
        
        # Map score to stress level (0-4)
        if stress_score < 2:
            stress_level = 0  # calm
        elif stress_score < 4:
            stress_level = 1  # low
        elif stress_score < 6:
            stress_level = 2  # moderate
        elif stress_score < 8:
            stress_level = 3  # high
        else:
            stress_level = 4  # extreme
        
        # Create probability distribution
        probs = np.zeros(5)
        probs[stress_level] = 0.7
        
        # Add neighboring probabilities
        if stress_level > 0:
            probs[stress_level - 1] = 0.2
        if stress_level < 4:
            probs[stress_level + 1] = 0.1
            
        return stress_level, probs, {
            'energy': float(energy),
            'avg_pitch': float(avg_pitch),
            'pitch_variation': float(pitch_variation),
            'speaking_rate': float(zcr),
            'spectral_centroid': float(spectral_centroid),
            'stress_score': float(stress_score)
        }
        
    except Exception as e:
        print(f"Voice analysis error: {e}")
        # Return moderate as default
        return 2, np.array([0.1, 0.2, 0.4, 0.2, 0.1]), {}

def extract_mfcc(audio_path, n_mfcc=40, target_length=87):
    """Extract MFCC features from audio (kept for compatibility)"""
    try:
        y, sr = librosa.load(audio_path, duration=3, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        if mfcc.shape[1] < target_length:
            pad_width = target_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :target_length]
        
        return mfcc
    except Exception as e:
        print(f"MFCC error: {e}")
        return np.zeros((n_mfcc, target_length))

STRESS_LEVELS = ['calm', 'low', 'moderate', 'high', 'extreme']