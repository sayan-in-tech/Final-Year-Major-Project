import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from models.face_model import FaceEmotionCNN, EMOTIONS

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for idx, emotion in enumerate(EMOTIONS):
            emotion_path = os.path.join(root_dir, emotion)
            if os.path.exists(emotion_path):
                for img_name in os.listdir(emotion_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(emotion_path, img_name))
                        self.labels.append(idx)
        
        print(f"Found {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            return self.__getitem__((idx + 1) % len(self))

def train_face_model():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Loading dataset...")
    dataset = FaceDataset('data/images', transform=transform)
    
    if len(dataset) == 0:
        print("ERROR: No images found! Check data/images/ folder structure")
        return
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print("Initializing model...")
    model = FaceEmotionCNN(num_classes=7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {device}...")
    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(dataloader)}, "
                      f"Loss: {running_loss/10:.4f}, Acc: {100*correct/total:.2f}%")
                running_loss = 0.0
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1} completed - Accuracy: {accuracy:.2f}%")
    
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/face_model.pth')
    print("✅ Model saved to checkpoints/face_model.pth")

if __name__ == '__main__':
    train_face_model()