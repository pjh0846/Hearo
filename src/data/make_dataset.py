import os
import pandas as pd
import numpy as np
import librosa
import torch
from tqdm import tqdm

# Configuration
DATA_DIR = r"c:\project\Hearo\data\raw\ESC-50"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_FILE = os.path.join(DATA_DIR, "meta", "esc50.csv")
PROCESSED_DIR = r"c:\project\Hearo\data\processed"

TARGET_CLASSES = {
    'siren': 0,
    'car_horn': 1,
    'glass_breaking': 2,
    'train': 3,
    'engine': 4,
    'door_wood_knock': 5,
    'crying_baby': 6,
    'dog': 7,
    'footsteps': 8,
    'clock_alarm': 9
}
OTHER_CLASS_LABEL = 10

def get_label(category):
    return TARGET_CLASSES.get(category, OTHER_CLASS_LABEL)

def extract_features(file_path, sr=22050, n_mels=128, duration=5):
    try:
        # Load audio
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Pad or truncate to fixed length
        target_length = sr * duration
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        else:
            y = y[:target_length]
            
        # Compute Mel-spectrogram
        mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mels_db = librosa.power_to_db(mels, ref=np.max)
        return mels_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    print("Loading metadata...")
    df = pd.read_csv(META_FILE)
    df['target_label'] = df['category'].apply(get_label)
    
    # Split based on folds
    # Folds 1, 2, 3 -> Train
    # Fold 4 -> Val
    # Fold 5 -> Test
    
    datasets = {
        'train': df[df['fold'].isin([1, 2, 3])],
        'val': df[df['fold'] == 4],
        'test': df[df['fold'] == 5]
    }
    
    for split_name, split_df in datasets.items():
        print(f"Processing {split_name} set ({len(split_df)} samples)...")
        X = []
        y = []
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            file_path = os.path.join(AUDIO_DIR, row['filename'])
            feature = extract_features(file_path)
            if feature is not None:
                X.append(feature)
                y.append(row['target_label'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Save processed data
        np.save(os.path.join(PROCESSED_DIR, f"X_{split_name}.npy"), X)
        np.save(os.path.join(PROCESSED_DIR, f"y_{split_name}.npy"), y)
        print(f"Saved {split_name}: X shape {X.shape}, y shape {y.shape}")

if __name__ == "__main__":
    main()
