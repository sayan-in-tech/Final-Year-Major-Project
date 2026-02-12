from transformers import pipeline
import torch

class TextSentimentAnalyzer:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        # Use a simpler sentiment pipeline
        self.model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device
        )
    
    def analyze(self, text):
        """Analyze sentiment and return probabilities for [negative, neutral, positive]"""
        result = self.model(text)[0]
        label = result['label']
        score = result['score']
        
        # Convert to 3-class format
        if label == 'POSITIVE':
            return [0.0, 0.0, score]  # positive
        else:  # NEGATIVE
            return [score, 0.0, 0.0]  # negative

SENTIMENTS = ['negative', 'neutral', 'positive']