# SELD Inference Module

Sound Event Localization and Detection (SELD) 추론을 위한 독립 모듈입니다.  
파일 또는 실시간 마이크 입력으로 소리 이벤트와 방향을 감지합니다.

## 요구사항

- Python 3.11.14 (use conda env)
- CUDA 지원 GPU (권장, CPU도 가능)
- 스테레오 오디오 입력 (2채널)

## 설치

```bash
pip install -r requirements.txt
```

## 입력 사양

| 항목 | 요구사항 |
|------|----------|
| **채널** | 스테레오 (2채널) 필수 |
| **샘플레이트** | 24,000 Hz (자동 리샘플링) |
| **포맷** | WAV, FLAC, MP3 등 librosa 지원 포맷 |
| **길이** | 최소 5초 이상 권장 (5초 윈도우 사용) |

### 실시간 입력
- 스테레오 마이크 필요 (2채널 이상)
- `--list-devices`로 사용 가능한 디바이스 확인

---

## 출력 형식

### 터미널 출력
```
[타임스탬프] 클래스명:방향°화살표 | 클래스명:방향°화살표 ...
```

**예시:**
```
[  5.0s] female_speech:+15°↑ | dog:-45°←
[  5.2s] crash:+66°→
[  5.4s] [No sound detected]
```

### 출력 설명
| 요소 | 설명 |
|------|------|
| `클래스명` | 감지된 소리 종류 (14개 클래스) |
| `+/-도` | 방위각 (Azimuth), 0°=정면, +는 오른쪽, -는 왼쪽 |
| `←` | 왼쪽 (-135° ~ -45°) |
| `↑` | 정면 (-45° ~ +45°) |
| `→` | 오른쪽 (+45° ~ +135°) |
| `↓` | 뒤쪽 (+135° 이상 또는 -135° 이하) |

### 중복 감지 통합
같은 클래스가 **15도 이내**에서 여러 번 감지되면 하나로 통합됩니다.

---

## 사용법

### 디바이스 목록 확인
```bash
python inference.py --list-devices
```

### 파일 추론
```bash
python inference.py --file path/to/audio.wav
```

### 실시간 마이크 추론
```bash
python inference.py
# 또는 특정 디바이스 사용
python inference.py --device 1
```

### 옵션
| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--file` | 오디오 파일 경로 | 없음 (마이크 사용) |
| `--device` | 오디오 디바이스 인덱스 | 기본 장치 |
| `--slide-ms` | 추론 간격 (ms) | 200 |
| `--weights` | 가중치 디렉토리 | `./weights` |

---

## 감지 가능한 소리 이벤트 (14개 클래스)

| ID | 클래스 | ID | 클래스 |
|----|--------|----|---------| 
| 0 | alarm | 7 | fire |
| 1 | baby | 8 | footsteps |
| 2 | crash | 9 | knock |
| 3 | dog | 10 | male_scream |
| 4 | engine | 11 | male_speech |
| 5 | female_scream | 12 | phone |
| 6 | female_speech | 13 | piano |

---

## 폴더 구조

```
seld_inference/
├── inference.py          # 메인 추론 스크립트
├── model.py              # SELDModel 아키텍처
├── inference_utils.py    # 유틸리티 함수
├── weights/
│   ├── best_model.pth    # 학습된 가중치
│   ├── config.pkl        # 모델 설정
│   └── scaler_dev.pkl    # 피처 정규화 스케일러
├── requirements.txt
└── README.md
```

---

## 다른 프로젝트에서 사용하기

```python
import sys
sys.path.append('/path/to/seld_inference')

from inference import RealtimeSELD

# 초기화
seld = RealtimeSELD(weights_dir='/path/to/seld_inference/weights')

# 파일 추론
seld.run_file('audio.wav')

# 또는 실시간 마이크 추론
seld.run_microphone()
```

---
