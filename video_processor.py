import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Tuple, Dict
from collections import Counter


def extract_frames(video_path: str, fps_sample: int = 1) -> List[Tuple[int, np.ndarray]]:
    """
    Extract frames from video at specified FPS sampling rate.
    
    Args:
        video_path: Path to video file
        fps_sample: Sample 1 frame every N seconds (default: 1 frame per second)
    
    Returns:
        List of tuples (frame_number, frame_array)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * fps_sample)  # frames to skip
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames at specified interval
        if frame_count % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame_count, frame_rgb))
        
        frame_count += 1
    
    cap.release()
    return frames


def detect_faces(frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in a frame using OpenCV Haar Cascade.
    
    Args:
        frame: RGB frame array
    
    Returns:
        List of (x, y, w, h) bounding boxes for detected faces
    """
    # Convert RGB to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Load Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces.tolist() if len(faces) > 0 else []


def process_video_emotions(
    video_path: str,
    face_model: torch.nn.Module,
    transform,
    device: torch.device,
    fps_sample: int = 1
) -> Dict:
    """
    Process video and detect emotions in faces.
    
    Args:
        video_path: Path to video file
        face_model: Loaded PyTorch face emotion model
        transform: Image preprocessing transform
        device: torch device (cpu/cuda)
        fps_sample: Sample 1 frame every N seconds
    
    Returns:
        Dictionary with:
        - emotions: List of detected emotions per frame
        - confidences: List of confidence scores
        - probabilities: List of probability distributions
        - frame_times: List of timestamps (in seconds)
        - dominant_emotion: Most common emotion
        - emotion_counts: Count of each emotion
        - avg_probabilities: Average probability distribution
    """
    # Extract frames
    frames = extract_frames(video_path, fps_sample)
    
    if len(frames) == 0:
        return {
            'emotions': [],
            'confidences': [],
            'probabilities': [],
            'frame_times': [],
            'dominant_emotion': None,
            'emotion_counts': {},
            'avg_probabilities': {}
        }
    
    # Get video FPS for timestamp calculation
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    all_emotions = []
    all_confidences = []
    all_probabilities = []
    frame_times = []
    
    face_model.eval()
    
    with torch.no_grad():
        for frame_num, frame in frames:
            # Calculate timestamp
            timestamp = frame_num / fps if fps > 0 else 0
            frame_times.append(timestamp)
            
            # Detect faces
            faces = detect_faces(frame)
            
            if len(faces) == 0:
                # No face detected, skip this frame
                all_emotions.append(None)
                all_confidences.append(0.0)
                all_probabilities.append(None)
                continue
            
            # Process first detected face (or could process all faces)
            x, y, w, h = faces[0]
            
            # Crop face region with some padding
            padding = 20
            y1 = max(0, y - padding)
            y2 = min(frame.shape[0], y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(frame.shape[1], x + w + padding)
            
            face_roi = frame[y1:y2, x1:x2]
            
            # Convert to PIL Image
            face_image = Image.fromarray(face_roi)
            
            # Preprocess
            face_tensor = transform(face_image).unsqueeze(0).to(device)
            
            # Predict
            output = face_model(face_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            emotion_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][emotion_idx].item()
            
            # Store results
            all_emotions.append(emotion_idx)
            all_confidences.append(confidence)
            all_probabilities.append(probs[0].cpu().numpy())
    
    # Aggregate results
    valid_emotions = [e for e in all_emotions if e is not None]
    
    if len(valid_emotions) == 0:
        return {
            'emotions': all_emotions,
            'confidences': all_confidences,
            'probabilities': all_probabilities,
            'frame_times': frame_times,
            'dominant_emotion': None,
            'emotion_counts': {},
            'avg_probabilities': {}
        }
    
    # Count emotions
    emotion_counts = Counter(valid_emotions)
    dominant_emotion_idx = emotion_counts.most_common(1)[0][0]
    
    # Calculate average probabilities
    valid_probs = [p for p in all_probabilities if p is not None]
    if len(valid_probs) > 0:
        avg_probabilities = np.mean(valid_probs, axis=0)
        avg_prob_dict = {i: float(avg_probabilities[i]) for i in range(len(avg_probabilities))}
    else:
        avg_prob_dict = {}
    
    return {
        'emotions': all_emotions,
        'confidences': all_confidences,
        'probabilities': all_probabilities,
        'frame_times': frame_times,
        'dominant_emotion': dominant_emotion_idx,
        'emotion_counts': dict(emotion_counts),
        'avg_probabilities': avg_prob_dict
    }
