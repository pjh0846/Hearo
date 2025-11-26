import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from model import LightweightCNN

# Configuration
PROCESSED_DIR = r"c:\project\Hearo\data\processed"
MODEL_PATH = r"c:\project\Hearo\models\hearo_model.pth"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_CLASSES = [
    'siren', 'car_horn', 'glass_breaking', 'train', 'engine',
    'door_wood_knock', 'crying_baby', 'dog', 'footsteps', 'clock_alarm', 'other'
]

class ESC50Dataset(Dataset):
    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)
        self.y = np.load(y_path)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X[idx]).unsqueeze(0)
        y = torch.LongTensor([self.y[idx]]).squeeze()
        return x, y

def test():
    print(f"Using device: {DEVICE}")
    
    # Load test data
    test_dataset = ESC50Dataset(
        os.path.join(PROCESSED_DIR, "X_test.npy"),
        os.path.join(PROCESSED_DIR, "y_test.npy")
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Load model
    model = LightweightCNN(num_classes=11).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    # Report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=TARGET_CLASSES))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

if __name__ == "__main__":
    test()
