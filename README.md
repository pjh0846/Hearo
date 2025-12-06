# 🎧 Hearo - 청각 장애인을 위한 소리 인식 시스템

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

## 📋 프로젝트 개요

**Hearo**는 청각 장애인을 위한 **실시간 소리 인식 및 분석 시스템**입니다.

딥러닝 기반의 소리 분류와 **Shazam 스타일 fingerprint 매칭** 기술을 결합하여:
- 🎯 **11가지 환경음** 자동 분류
- 🔔 **사용자 정의 소리** 인식 (초인종, 알람 등 개인 맞춤)
- 📊 **긴급도, 세기, 거리** 등 소리 특징 분석

## ✨ 주요 기능

### 1. 🎯 소리 종류 분류 (11개 클래스)

| 카테고리 | 소리 종류 |
|---------|----------|
| 🚨 위험 | 사이렌, 자동차 경적, 유리 깨지는 소리 |
| 🚗 교통 | 기차, 엔진 |
| 🏠 생활 | 문 노크, 발소리, 알람 |
| 👶 사람/동물 | 아기 울음, 개 짖는 소리 |
| ❓ 기타 | 미분류 소리 |

### 2. 🔔 커스텀 소리 탐지 (Shazam 스타일)

사용자가 직접 녹음한 소리를 등록하여 인식:
- **초인종** - 집 초인종, 아파트 초인종 등
- **가전제품** - 전자레인지, 세탁기 알림 등
- **개인 알람** - 사용자 맞춤 알림음

> 💡 `templates/` 폴더에 `.wav` 파일을 추가하면 자동으로 인식됩니다!

### 3. 📊 4가지 소리 특징 분석

| 특징 | 설명 | 분석 방법 |
|-----|------|----------|
| **소리 종류** | 11개 클래스 분류 | CNN + Mel-spectrogram |
| **긴급도** | 높음/보통/낮음 | 소리 종류(60%) + 음향 특징(40%) |
| **세기** | 상/중/하 | RMS 에너지 기반 |
| **거리** | 가까움/보통/멂 | 고주파 감쇠 패턴 분석 |

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/pjh0846/Hearo.git
cd Hearo

# 패키지 설치
pip install torch torchvision torchaudio librosa numpy pandas scikit-learn flask flask-cors

# (선택) 비디오 파일 지원
pip install moviepy
```

### 실행 방법

#### 1️⃣ 웹 UI 실행
```bash
cd ui
python app.py
# 브라우저에서 http://localhost:5000 접속
```

#### 2️⃣ 명령줄 테스트
```bash
# 기본 분석
python test_audio_features.py <파일경로>

# 세부 정보 표시
python test_audio_features.py <파일경로> --details

# 예시
python test_audio_features.py data/raw/ESC-50/audio/1-100032-A-0.wav
```

### 출력 예시

```
============================================================
🎵  Hearo - 소리 특징 분석 결과
============================================================

🔊 소리 종류: 사이렌 (siren)
   신뢰도: 80.88%

🟡 긴급도: 보통 (0.66)
   ├─ 기본 가중치: 0.95
   ├─ 음향 긴급도: 0.23
   │  ├─ 주파수 중심: 0.09
   │  ├─ 변화율: 0.50
   │  ├─ 급격한 변화: 0.10
   │  └─ 빠르기: 0.33

📊 세  기: 상 (0.76) ███████
📌 거  리: 보통 (0.37)

============================================================

💬 요약: 보통 긴급도의 상 크기의 사이렌 소리가 보통에서 감지되었습니다.
```

## 📁 프로젝트 구조

```
Hearo/
├── 📂 data/
│   ├── raw/                    # 원본 ESC-50 데이터셋
│   ├── processed/              # 전처리된 학습 데이터
│   └── classified/             # 클래스별 분류 데이터
│
├── 📂 src/
│   ├── config.py               # 설정 파일
│   ├── data/
│   │   └── make_dataset.py     # 데이터셋 생성
│   ├── features/
│   │   └── audio_features.py   # 오디오 특징 추출
│   ├── models/
│   │   ├── model.py            # LightweightCNN 모델 정의
│   │   ├── train_model.py      # 모델 학습
│   │   ├── test_model.py       # 모델 평가
│   │   ├── predict.py          # 예측 및 특징 분석
│   │   └── custom_detector.py  # 커스텀 소리 탐지 (Fingerprint)
│   └── utils/
│       └── organize_data.py    # 데이터 정리 유틸
│
├── 📂 models/
│   └── hearo_model.pth         # 학습된 모델 가중치
│
├── 📂 templates/               # 🔔 커스텀 소리 템플릿
│   ├── doorbellA.wav           # 초인종 소리들
│   ├── doorbellB.wav
│   ├── microwave.wav           # 전자레인지 알림
│   ├── LG.wav                  # LG 알림음
│   └── SAMSUNG.wav             # 삼성 알림음
│
├── 📂 ui/                      # 웹 UI
│   ├── app.py                  # Flask 백엔드
│   ├── index.html              # 메인 페이지
│   └── static/
│       ├── style.css           # 스타일시트
│       └── script.js           # 프론트엔드 로직
│
└── README.md
```

## 🏗️ 모델 아키텍처

### LightweightCNN

```
Input: Mel-spectrogram (128 × time)
    ↓
Conv2D (32) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D (64) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D (128) → BatchNorm → ReLU → MaxPool
    ↓
Conv2D (256) → BatchNorm → ReLU → MaxPool
    ↓
Global Average Pooling
    ↓
Dropout (0.5)
    ↓
FC → 11 classes
```

### 커스텀 소리 탐지 (Fingerprint Matching)

```
Template Audio           Input Audio
     ↓                        ↓
 STFT Spectrogram       STFT Spectrogram
     ↓                        ↓
 Peak Detection         Peak Detection
     ↓                        ↓
 Fingerprint Keys       Fingerprint Keys
     (f1, f2, Δt)           (f1, f2, Δt)
           ↘                   ↙
              Set Intersection
                    ↓
              Similarity Score
```

## 📊 데이터셋

- **ESC-50**: Environmental Sound Classification
  - 50개 클래스, 2000개 오디오 클립 (각 5초)
  - 본 프로젝트: 청각 장애인 관련 **10개 클래스 + 기타** 사용

## 🛠️ 학습하기

```bash
# 1. 데이터 전처리
cd src/data
python make_dataset.py

# 2. 모델 학습
cd ../models
python train_model.py

# 3. 모델 평가
python test_model.py
```

## 🔔 커스텀 소리 등록하기

1. `templates/` 폴더에 `.wav` 파일 추가
2. 파일명이 소리 라벨로 사용됨 (예: `doorbell.wav` → "doorbell")
3. 서버 재시작 시 자동 로드

```bash
# 예시: 초인종 소리 등록
cp my_doorbell.wav templates/초인종.wav
```

## 🎬 지원 파일 형식

| 구분 | 확장자 |
|-----|-------|
| 오디오 | `.wav`, `.mp3`, `.flac` |
| 비디오 | `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv` |

> 비디오 파일은 자동으로 오디오가 추출되어 분석됩니다.

## 🔧 기술 스택

| 분야 | 기술 |
|-----|-----|
| Deep Learning | PyTorch |
| Audio Processing | Librosa |
| Web Backend | Flask, Flask-CORS |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Video Processing | MoviePy (선택) |

## 📈 향후 계획

- [ ] 실시간 마이크 입력 지원
- [ ] 모바일 앱 개발 (React Native)
- [ ] 다중 소리 동시 감지
- [ ] 소리 방향 감지 (스테레오/마이크 배열)
- [ ] 사용자 맞춤 긴급도 설정
- [ ] 진동/시각적 알림 연동

## 📄 라이선스

MIT License

---

<p align="center">
  Made with ❤️ for accessibility
</p>
