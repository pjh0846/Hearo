import numpy as np
import os

PROCESSED_DIR = r"c:\project\Hearo\data\processed"

def check_distribution(split):
    y_path = os.path.join(PROCESSED_DIR, f"y_{split}.npy")
    if not os.path.exists(y_path):
        print(f"{split} not found")
        return
        
    y = np.load(y_path)
    unique, counts = np.unique(y, return_counts=True)
    print(f"--- {split} distribution ---")
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c}")

check_distribution('train')
check_distribution('val')
