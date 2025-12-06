import os
import sys
import numpy as np
import torch
import librosa
import tempfile

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import MODEL_PATH, DEVICE, SAMPLE_RATE, N_MELS, DURATION
from config import CUSTOM_TEMPLATE_DIR, CUSTOM_DETECTION_THRESHOLD
from models.model import LightweightCNN
from models.custom_detector import try_custom_detection
from features.audio_features import extract_all_features


def load_model(model_path=MODEL_PATH):
    """
    학습된 모델 로드
    
    Args:
        model_path: 모델 파일 경로
    
    Returns:
        model: 로드된 PyTorch 모델
    """
    model = LightweightCNN(num_classes=11).to(DEVICE)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"✅ 모델 로드 완료: {model_path}")
    else:
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    return model


def extract_audio_from_video(file_path):
    """
    비디오 파일에서 오디오 추출
    """
    try:
        # moviepy v2 import 방식
        try:
            from moviepy import VideoFileClip
        except ImportError:
            # moviepy v1 import 방식 (호환성)
            from moviepy.editor import VideoFileClip
        
        print(f"🎬 비디오 파일에서 오디오 추출 중...")
        
        # 임시 오디오 파일 생성
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        # 비디오에서 오디오 추출
        video = VideoFileClip(file_path)
        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        video.close()
        
        print(f"✅ 오디오 추출 완료")
        return temp_audio_path, True
        
    except ImportError:
        raise ImportError(
            "MP4 파일 처리를 위해 moviepy가 필요합니다.\n"
            "설치: pip install moviepy"
        )


def preprocess_audio(file_path, sr=SAMPLE_RATE, n_mels=N_MELS, duration=DURATION):
    """
    오디오 파일 전처리 (모델 입력 형식으로 변환)
    MP4 비디오 파일의 경우 오디오를 자동으로 추출
    
    Args:
        file_path: 오디오/비디오 파일 경로 (.wav, .mp3, .mp4 등)
        sr: 샘플링 레이트
        n_mels: Mel-spectrogram의 주파수 밴드 수
        duration: 오디오 길이 (초)
    
    Returns:
        mels_db: Mel-spectrogram (dB)
        audio: 원본 오디오 신호
        sr: 샘플링 레이트
    """
    
    # MP4 파일인 경우 오디오 추출
    file_ext = os.path.splitext(file_path)[1].lower()
    temp_audio_path = None
    is_temp = False
    
    if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
        audio_path_to_load, is_temp = extract_audio_from_video(file_path)
        temp_audio_path = audio_path_to_load
    else:
        audio_path_to_load = file_path
    
    try:
        # 오디오 로드
        audio, _ = librosa.load(audio_path_to_load, sr=sr, duration=duration)
        
        # 고정 길이로 패딩/자르기
        target_length = sr * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Mel-spectrogram 계산
        mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mels_db = librosa.power_to_db(mels, ref=np.max)
        
        return mels_db, audio, sr
        
    finally:
        # 임시 파일 정리
        if is_temp and temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except:
                pass


def predict_sound_class(model, mels_db, confidence_threshold=0.6):
    """
    소리 종류 예측
    
    Args:
        model: PyTorch 모델
        mels_db: Mel-spectrogram (dB)
        confidence_threshold: 신뢰도 임계값 (기본값: 0.6)
                            이 값 미만이면 "기타(other)" 클래스(10)로 분류
    
    Returns:
        predicted_class: 예측된 클래스 인덱스
        confidence: 예측 확률
    """
    # Tensor로 변환: (1, 1, n_mels, time)
    x = torch.FloatTensor(mels_db).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # 예측
    with torch.no_grad():
        outputs = model(x)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    predicted_class = int(predicted.item())
    confidence_value = float(confidence.item())
    
    # Confidence Threshold 적용
    # 신뢰도가 낮으면 "기타" 클래스(10)로 분류
    if predicted_class != 10 and confidence_value < confidence_threshold:
        predicted_class = 10  # "기타(other)" 클래스
    
    return predicted_class, confidence_value


def predict_audio_features(file_path, model=None):
    """
    오디오 파일에서 모든 특징 예측
    
    Args:
        file_path: 오디오 파일 경로
        model: 미리 로드된 모델 (None이면 새로 로드)
    
    Returns:
        dict: 모든 특징 정보
    """
    # 모델 로드 (없으면)
    if model is None:
        model = load_model()
    
    # 오디오 전처리
    mels_db, audio, sr = preprocess_audio(file_path)
    
    # 1순위: 커스텀 소리 탐지 시도
    custom_label, custom_score = try_custom_detection(
        audio, sr, template_dir=CUSTOM_TEMPLATE_DIR
    )
    if custom_label:
        # 커스텀 소리로 결과 생성 (분류 없이 바로 반환)
        return build_custom_result(custom_label, custom_score, audio, sr, custom_score)
    
    # 2순위: 모델 분류 (커스텀 탐지 실패 시)
    predicted_class, confidence = predict_sound_class(model, mels_db)
    
    # 모든 특징 추출
    features = extract_all_features(audio, sr, predicted_class)
    
    # 신뢰도 추가
    features['confidence'] = confidence
    
    return features


def build_custom_result(label: str, score: float, audio, sr, original_confidence: float):
    """
    커스텀 소리 탐지 결과 생성
    
    Args:
        label: 커스텀 소리 라벨 (예: "doorbellA", "LG")
        score: 탐지 점수
        audio: 오디오 신호
        sr: 샘플링 레이트
        original_confidence: 원래 분류 모델의 신뢰도
    
    Returns:
        dict: 커스텀 소리 특징 정보
    """
    from features.audio_features import extract_intensity, estimate_distance
    
    # 음향 특징 추출
    intensity = extract_intensity(audio, sr)
    distance = estimate_distance(audio, sr)
    
    # 커스텀 소리는 중간 긴급도로 설정 (사용자 정의 소리)
    urgency = {
        'level': '보통',
        'value': 0.5,
        'components': {
            'base_urgency': 0.5,
            'custom_detection': True
        }
    }
    
    return {
        'sound_type': {
            'name_en': f'custom:{label}',
            'name_kr': f'사용자 등록: {label}',
            'class_idx': -1,  # 커스텀 클래스
            'custom_label': label,
            'custom_score': score
        },
        'intensity': intensity,
        'distance': distance,
        'urgency': urgency,
        'confidence': original_confidence,
        'is_custom_detection': True
    }

