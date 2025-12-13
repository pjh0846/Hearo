# 🎧 Hearo - SELD 기반 소리 감지 시스템

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

## 📋 프로젝트 개요

**Hearo**는 청각 장애인을 위한 **실시간 소리 인식 및 방향 감지 시스템**입니다.

SELD(Sound Event Localization and Detection) 기술과 **Shazam 스타일 fingerprint 매칭**을 결합하여:
- 🎯 **14가지 환경음** 실시간 분류
- 📍 **소리 방향 감지** (360도 방위각)
- 🔔 **사용자 정의 소리** 인식 (초인종, 알람 등 개인 맞춤)
- 📊 **스테레오 오디오** 분석

## ✨ 주요 기능

### 1. 🎯 SELD - 소리 종류 및 방향 감지

| 카테고리 | 소리 종류 |
|---------|----------|
| 🚨 위험 | 충돌음, 화재, 비명 (남성/여성) |
| 🏠 생활 | 문 노크, 발소리, 알람, 전화 |
| 👶 사람/동물 | 아기 울음, 개 짖는 소리, 말소리 (남성/여성) |
| 🎵 기타 | 피아노, 엔진 |

**총 14개 클래스** - 각 소리의 방향(방위각)까지 실시간 감지

### 2. 📍 방향 감지

스테레오 오디오 분석을 통해 소리의 방향을 4방향으로 표시:
- **정면** (↑) : -45° ~ 45°
- **왼쪽** (←) : -135° ~ -45°
- **오른쪽** (→) : 45° ~ 135°
- **뒤쪽** (↓) : 135° ~ 180° / -180° ~ -135°

### 3. 🔔 커스텀 소리 탐지

사용자가 직접 녹음한 소리를 등록하여 인식:
- **초인종** - 집 초인종, 아파트 초인종 등
- **가전제품** - 전자레인지, 세탁기 알림 등
- **개인 알람** - 사용자 맞춤 알림음

> 💡 `templates/` 폴더에 `.wav` 파일을 추가하면 자동으로 인식됩니다!

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/pjh0846/Hearo.git
cd Hearo

# 패키지 설치
pip install torch torchvision torchaudio librosa numpy scipy flask flask-cors
```

### 실행 방법

#### 웹 UI 실행
```bash
cd web
python app.py
# 브라우저에서 http://localhost:5000 접속
```

## 📁 프로젝트 구조

```
Hearo/
├── 📂 backend/                 # 백엔드 코드
│   ├── seld/                   # SELD 모델 추론 엔진
│   │   ├── inference.py        # 메인 추론 로직
│   │   ├── inference_utils.py  # 오디오 처리 유틸
│   │   ├── model.py            # SELD 모델 정의
│   │   └── weights/            # 학습된 가중치
│   ├── custom/                 # 커스텀 소리 탐지
│   │   └── custom_detector.py  # Fingerprint 기반 탐지
│   └── config.py               # 설정 파일
│
├── 📂 web/                     # 웹 UI
│   ├── app.py                  # Flask 백엔드
│   ├── index.html              # 메인 페이지
│   └── static/
│       ├── style.css           # 스타일시트
│       └── script.js           # 프론트엔드 로직
│
├── 📂 templates/               # 🔔 커스텀 소리 템플릿
├── 📂 data/                    # 데이터셋
│
└── README.md
```

## 🏗️ SELD 아키텍처

### SELD 모델 (Sound Event Localization and Detection)

```
Stereo Audio Input (L/R channels)
        ↓
Multi-head Feature Extraction
    ↓               ↓
SED Branch      DOA Branch
(분류)           (방향 추정)
    ↓               ↓
14 Classes    Azimuth (방위각)
```

**특징:**
- 스테레오 입력으로 좌/우 채널 정보 활용
- Multi-task learning으로 분류와 방향을 동시 학습
- DCASE2020 챌린지 기반 모델

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
| 오디오 | `.wav` (스테레오 권장) |

> ⚠️ **중요**: SELD 모델은 **스테레오 오디오**가 필요합니다. 모노 오디오는 방향 감지가 불가능합니다.

## 🔧 기술 스택

| 분야 | 기술 |
|-----|-----|
| Deep Learning | PyTorch |
| Audio Processing | Librosa, SciPy |
| Web Backend | Flask, Flask-CORS |
| SELD Model | DCASE2020 기반 |

## 📈 향후 계획

- [ ] 실시간 마이크 입력 지원
- [ ] 모바일 앱 개발
- [ ] 다중 소리 동시 감지 개선
- [ ] 고도(elevation) 감지 추가
- [ ] 사용자 맞춤 긴급도 설정
- [ ] 진동/시각적 알림 연동

## 📄 라이선스

MIT License

---

<p align="center">
  Made with ❤️ for accessibility
</p>
