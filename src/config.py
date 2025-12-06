import os
import torch

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hearo_model.pth")

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Audio Configuration
SAMPLE_RATE = 22050
N_MELS = 128
DURATION = 5

# Custom Detection Configuration
CUSTOM_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")
CUSTOM_DETECTION_THRESHOLD = 0.15
CUSTOM_DETECTION_SR = 16000

# Urgency Rules (0.0 ~ 1.0)
URGENCY_RULES = {
    'siren': 0.95,           # 사이렌 - 매우 높음
    'car_horn': 0.85,        # 자동차 경적 - 높음
    'glass_breaking': 0.90,  # 유리 깨짐 - 매우 높음
    'train': 0.50,           # 기차 - 보통
    'engine': 0.40,          # 엔진 - 낮음
    'door_wood_knock': 0.30, # 문 노크 - 낮음
    'crying_baby': 0.75,     # 아기 울음 - 높음
    'dog': 0.60,             # 개 짖음 - 보통
    'footsteps': 0.20,       # 발소리 - 낮음
    'clock_alarm': 0.70,     # 알람 - 높음
    'other': 0.50            # 기타 - 보통
}

# Class Names Mapping
CLASS_NAMES = {
    0: 'siren',
    1: 'car_horn',
    2: 'glass_breaking',
    3: 'train',
    4: 'engine',
    5: 'door_wood_knock',
    6: 'crying_baby',
    7: 'dog',
    8: 'footsteps',
    9: 'clock_alarm',
    10: 'other'
}

# Korean Class Names Mapping
CLASS_NAMES_KR = {
    'siren': '사이렌',
    'car_horn': '자동차 경적',
    'glass_breaking': '유리 깨지는 소리',
    'train': '기차',
    'engine': '엔진',
    'door_wood_knock': '문 노크',
    'crying_baby': '아기 울음',
    'dog': '개 짖는 소리',
    'footsteps': '발소리',
    'clock_alarm': '알람',
    'other': '기타'
}
