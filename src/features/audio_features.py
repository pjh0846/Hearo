import numpy as np
import librosa
import os
import sys

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import URGENCY_RULES, CLASS_NAMES, CLASS_NAMES_KR


def extract_intensity(audio, sr=22050):
    """
    소리의 세기(음량) 추출
    
    Args:
        audio: 오디오 신호 (numpy array)
        sr: 샘플링 레이트
    
    Returns:
        dict: {'level': '상/중/하', 'value': float (0~1)}
    """
    # RMS (Root Mean Square) 에너지 계산
    rms = librosa.feature.rms(y=audio)[0]
    avg_rms = np.mean(rms)
    
    # dB로 변환 (정규화)
    rms_db = 20 * np.log10(avg_rms + 1e-8)
    
    # -60dB ~ 0dB 범위로 정규화 (0~1)
    normalized = np.clip((rms_db + 60) / 60, 0, 1)
    
    # 3단계로 분류
    if normalized > 0.65:
        level = '상'
    elif normalized > 0.35:
        level = '중'
    else:
        level = '하'
    
    return {
        'level': level,
        'value': float(normalized)
    }


def estimate_distance(audio, sr=22050):
    """
    소리의 거리 추정
    
    Args:
        audio: 오디오 신호 (numpy array)
        sr: 샘플링 레이트
    
    Returns:
        dict: {'level': '가까움/보통/멂', 'value': float (0~1)}
    """
    # 1. RMS 에너지 (음량 기반)
    rms = librosa.feature.rms(y=audio)[0].mean()
    
    # 2. Spectral Centroid (주파수 중심)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()
    
    # 3. Spectral Rolloff (고주파 성분)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0].mean()
    
    # 정규화
    rms_normalized = np.clip(rms * 50, 0, 1)  # RMS는 낮은 값
    centroid_normalized = np.clip(spectral_centroid / sr, 0, 1)
    rolloff_normalized = np.clip(spectral_rolloff / sr, 0, 1)
    
    # 멀수록 고주파가 약해지고 음량이 작음
    # 거리 점수: 높을수록 멈 (0: 가까움, 1: 멂)
    distance_score = 1 - (
        rms_normalized * 0.6 +           # 음량 (낮으면 멂)
        centroid_normalized * 0.2 +      # 중심 주파수
        rolloff_normalized * 0.2         # 고주파 성분
    )
    
    # 3단계로 분류
    if distance_score < 0.35:
        level = '가까움'
    elif distance_score < 0.65:
        level = '보통'
    else:
        level = '멂'
    
    return {
        'level': level,
        'value': float(distance_score)
    }


def get_urgency(sound_class_name, audio, sr):
    """
    소리 종류와 음향 특징을 결합하여 긴급도 계산
    
    Args:
        sound_class_name: 소리 종류 이름 (영문)
        audio: 오디오 신호 (numpy array)
        sr: 샘플링 레이트
    
    Returns:
        dict: {'level': '높음/보통/낮음', 'value': float (0~1), 'components': dict}
    """
    # 1. 소리 종류별 기본 가중치
    base_urgency = URGENCY_RULES.get(sound_class_name, 0.5)
    
    # 2. 음향 특징 기반 긴급도 계산
    
    # 2-1. Spectral Centroid (주파수 중심 - 높을수록 날카롭고 긴급)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0].mean()
    centroid_normalized = np.clip(spectral_centroid / (sr / 2), 0, 1)  # 0~1 정규화
    
    # 2-2. Zero Crossing Rate (빠른 변화율 - 높을수록 날카롭고 긴급)
    zcr = librosa.feature.zero_crossing_rate(audio)[0].mean()
    zcr_normalized = np.clip(zcr * 10, 0, 1)  # 0~1 정규화
    
    # 2-3. Onset Strength (급격한 에너지 변화 - 높을수록 긴급)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_strength = np.max(onset_env)
    onset_normalized = np.clip(onset_strength / 40, 0, 1)  # 0~1 정규화
    
    # 2-4. Tempo (빠르기 - 빠를수록 긴급)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_normalized = np.clip((tempo - 60) / 120, 0, 1)  # 60~180 BPM 범위
    
    # 3. 음향 특징 종합 점수 (가중 평균)
    acoustic_urgency = (
        centroid_normalized * 0.35 +   # 주파수 중심 (35%)
        zcr_normalized * 0.25 +         # 변화율 (25%)
        onset_normalized * 0.25 +       # 급격한 변화 (25%)
        tempo_normalized * 0.15         # 빠르기 (15%)
    )
    
    # 4. 최종 긴급도 = 기본 가중치(60%) + 음향 특징(40%)
    final_urgency = base_urgency * 0.6 + acoustic_urgency * 0.4
    final_urgency = np.clip(final_urgency, 0, 1)
    
    # 3단계로 분류
    if final_urgency > 0.7:
        level = '높음'
    elif final_urgency > 0.4:
        level = '보통'
    else:
        level = '낮음'
    
    return {
        'level': level,
        'value': float(final_urgency),
        'components': {
            'base_urgency': float(base_urgency),
            'acoustic_urgency': float(acoustic_urgency),
            'spectral_centroid': float(centroid_normalized),
            'zero_crossing_rate': float(zcr_normalized),
            'onset_strength': float(onset_normalized),
            'tempo': float(tempo_normalized)
        }
    }


def extract_all_features(audio, sr, predicted_class_idx):
    """
    모든 특징 추출 (종류, 긴급도, 세기, 거리)
    
    Args:
        audio: 오디오 신호 (numpy array)
        sr: 샘플링 레이트
        predicted_class_idx: 예측된 소리 종류 인덱스
    
    Returns:
        dict: 모든 특징이 포함된 딕셔너리
    """
    # 소리 종류 이름
    sound_class_name = CLASS_NAMES.get(predicted_class_idx, 'other')
    sound_class_name_kr = CLASS_NAMES_KR.get(sound_class_name, '기타')
    
    # 각 특징 추출
    intensity = extract_intensity(audio, sr)
    distance = estimate_distance(audio, sr)
    urgency = get_urgency(sound_class_name, audio, sr)  # audio, sr 추가
    
    return {
        'sound_type': {
            'name_en': sound_class_name,
            'name_kr': sound_class_name_kr,
            'class_idx': int(predicted_class_idx)
        },
        'intensity': intensity,
        'distance': distance,
        'urgency': urgency
    }
