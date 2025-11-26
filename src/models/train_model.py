import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from model import LightweightCNN

# Configuration
PROCESSED_DIR = r"c:\project\Hearo\data\processed"
MODEL_SAVE_PATH = r"c:\project\Hearo\models\hearo_model.pth"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ESC50Dataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Add channel dimension: (1, n_mels, time)
        x = torch.FloatTensor(self.X[idx]).unsqueeze(0)
        y = torch.LongTensor([self.y[idx]]).squeeze()
        return x, y

def train():
    print(f"Using device: {DEVICE}")
    
    # Load data
    train_dataset = ESC50Dataset(
        os.path.join(PROCESSED_DIR, "X_train.npy"),
        os.path.join(PROCESSED_DIR, "y_train.npy")
    )
    val_dataset = ESC50Dataset(
        os.path.join(PROCESSED_DIR, "X_val.npy"),
        os.path.join(PROCESSED_DIR, "y_val.npy")
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Calculate class weights
    # Count samples per class in train_dataset
    y_train = train_dataset.y
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    num_classes = len(class_counts)
    
    # weight = total / (num_classes * count)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights = torch.FloatTensor(class_weights).to(DEVICE)
    
    print(f"Class weights: {class_weights}")
    
    # Model
    model = LightweightCNN(num_classes=11).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH))
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Model saved!")

if __name__ == "__main__":
    train()
