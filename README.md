# Hearo - 청각 장애인을 위한 소리 인식 시스템

## 프로젝트 개요
Hearo는 청각 장애인을 위한 실시간 소리 인식 및 분석 시스템입니다.
딥러닝 기반의 소리 분류와 음향 신호 처리 기법을 결합하여 다양한 소리를 감지하고 특징을 추출합니다.

## 주요 기능

### 🎯 4가지 소리 특징 추출
1. **소리 종류 분류** (11개 클래스)
   - 사이렌, 자동차 경적, 유리 깨지는 소리, 기차, 엔진
   - 문 노크, 아기 울음, 개 짖는 소리, 발소리, 알람, 기타

2. **긴급도 분석** (높음/보통/낮음)
   - 소리 종류 기본 가중치 (60%)
   - 음향 특징 분석 (40%)
     - 주파수 중심 (Spectral Centroid)
     - 변화율 (Zero Crossing Rate)
     - 급격한 변화 (Onset Strength)
     - 빠르기 (Tempo)

3. **세기/음량 측정** (상/중/하)
   - RMS 에너지 기반 음량 측정
   - 0~1 스케일 정규화

4. **거리 추정** (가까움/보통/멂)
   - RMS, Spectral Centroid, Spectral Rolloff 결합
   - 고주파 감쇠 패턴 분석

### 🎬 지원 파일 형식
- 오디오: `.wav`, `.mp3`, `.flac`
- 비디오: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv` (자동 오디오 추출)

## 사용 방법

### 설치
```bash
# 기본 패키지 설치
pip install torch torchvision torchaudio librosa numpy pandas scikit-learn

# 비디오 파일 지원 (선택)
pip install moviepy
```

### 오디오/비디오 분석
```bash
# 기본 사용
python test_audio_features.py <파일경로>

# 세부 정보 표시
python test_audio_features.py <파일경로> --details

# 예시
python test_audio_features.py data/raw/ESC-50/audio/1-100032-A-0.wav
python test_audio_features.py video.mp4 --details
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

## 프로젝트 구조
```
Hearo/
├── data/
│   ├── raw/              # 원본 데이터 (ESC-50)
│   └── processed/        # 전처리된 데이터
├── src/
│   ├── data/            # 데이터 처리
│   ├── features/        # 특징 추출
│   │   └── audio_features.py
│   └── models/          # 모델 정의 및 예측
│       ├── model.py
│       ├── train_model.py
│       ├── test_model.py
│       └── predict.py
├── models/              # 학습된 모델 저장
│   └── hearo_model.pth
├── test_audio_features.py  # 테스트 스크립트
└── README.md
```

## 모델 아키텍처
- **LightweightCNN**: 경량화된 CNN 모델
  - 4개의 Convolutional Layers + BatchNorm + MaxPooling
  - Global Average Pooling
  - Dropout (0.5)
  - 입력: Mel-spectrogram (128 x time)
  - 출력: 11개 클래스

## 데이터셋
- **ESC-50**: Environmental Sound Classification
  - 50개 클래스의 환경 소리
  - 2000개의 5초 오디오 클립
  - 프로젝트에서는 청각 장애인 관련 10개 클래스 + 기타 사용

## 학습
```bash
# 데이터 전처리
cd src/data
python make_dataset.py

# 모델 학습
cd ../models
python train_model.py

# 모델 평가
python test_model.py
```

## 기술 스택
- **PyTorch**: 딥러닝 프레임워크
- **Librosa**: 오디오 신호 처리
- **NumPy/Pandas**: 데이터 처리
- **MoviePy**: 비디오 오디오 추출 (선택)

## 향후 계획
- [ ] 실시간 소리 감지
- [ ] 모바일 앱 개발
- [ ] 다중 소리 동시 감지
- [ ] 소리 방향 감지 (스테레오/마이크 배열)
- [ ] 사용자 맞춤 긴급도 설정

## 라이선스
MIT License
