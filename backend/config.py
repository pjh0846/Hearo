"""
Hearo SELD Configuration
SELD 모델 및 커스텀 소리 탐지 설정
"""

import os

# ============================================================
# Base Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# ============================================================
# Custom Detection Configuration
# ============================================================

CUSTOM_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")
CUSTOM_DETECTION_THRESHOLD = 0.15
CUSTOM_DETECTION_SR = 16000
