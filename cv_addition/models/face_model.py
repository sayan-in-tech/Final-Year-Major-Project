import torch
import torch.nn as nn
import torchvision.models as models

class FaceEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FaceEmotionCNN, self).__init__()
        # Use MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        # Modify classifier
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.mobilenet.last_channel, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)

# Emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']