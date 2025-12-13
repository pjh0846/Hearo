"""
커스텀 소리 탐지 모듈

Shazam 스타일 fingerprint 기반으로 사용자가 등록한 소리를 탐지합니다.
분류 모델이 "기타(other)"로 분류한 경우에만 실행됩니다.
"""

import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import librosa


# ==========================
# 1. 유틸: 오디오 로딩 + 템플릿 자르기
# ==========================

def load_mono(audio_path: str, sr: int = 16000) -> np.ndarray:
    """오디오를 mono + 지정 sr로 로드."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    return y.astype(np.float32)


def crop_loudest_segment(y: np.ndarray, sr: int, max_sec: float = 2.0) -> np.ndarray:
    """
    긴 템플릿에서 에너지가 가장 큰 max_sec 구간만 사용.
    """
    max_len = int(sr * max_sec)
    n = len(y)
    if n <= max_len:
        return y

    # 100ms 단위로 에너지 계산
    frame_len = int(0.1 * sr)
    hop = frame_len // 2
    if frame_len <= 0:
        return y[:max_len]

    energies = []
    starts = []
    for start in range(0, n - frame_len + 1, hop):
        seg = y[start:start + frame_len]
        e = float(np.mean(seg**2))
        energies.append(e)
        starts.append(start)

    if not energies:
        return y[:max_len]

    best_start_frame = starts[int(np.argmax(energies))]
    center = best_start_frame + frame_len // 2

    start = max(0, center - max_len // 2)
    end = min(n, start + max_len)
    return y[start:end]


# ==========================
# 2. Fingerprint 추출
# ==========================

def compute_spectrogram(
    y: np.ndarray,
    sr: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT 기반 magnitude spectrogram 계산."""
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_mag = np.abs(S)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(
        np.arange(S_mag.shape[1]), sr=sr, hop_length=hop_length
    )
    return S_mag, freqs, times


def find_spectral_peaks(
    S_mag: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    top_k_per_frame: int = 8,
    min_mag_db: float = -45.0,
) -> List[Tuple[int, int]]:
    """각 프레임별 magnitude 상위 top_k_per_frame 주파수 bin을 peak으로 사용."""
    S_db = librosa.amplitude_to_db(S_mag + 1e-10, ref=np.max)

    n_freqs, n_frames = S_db.shape
    peaks: List[Tuple[int, int]] = []

    for t in range(n_frames):
        col = S_db[:, t]
        valid_idx = np.where(col > min_mag_db)[0]
        if len(valid_idx) == 0:
            continue

        # 상위 top_k_per_frame만 사용
        top_idx = np.argsort(col[valid_idx])[-top_k_per_frame:]
        for idx in top_idx:
            f_idx = valid_idx[idx]
            peaks.append((t, f_idx))

    return peaks


def build_fingerprint_keys(
    y: np.ndarray,
    sr: int = 16000,
    fan_out: int = 7,
    min_delta_t: int = 1,
    max_delta_t: int = 30,
) -> List[Tuple[int, int, int]]:
    """
    Shazam 스타일 fingerprint key 생성.
    (f1, f2, dt)를 hash key로 사용.
    """
    S_mag, freqs, times = compute_spectrogram(y, sr=sr)
    peaks = find_spectral_peaks(S_mag, freqs, times)

    peaks = sorted(peaks, key=lambda x: x[0])
    keys: List[Tuple[int, int, int]] = []

    n_peaks = len(peaks)
    for i in range(n_peaks):
        t1, f1 = peaks[i]
        for j in range(1, fan_out + 1):
            if i + j >= n_peaks:
                break
            t2, f2 = peaks[i + j]
            dt = t2 - t1
            if dt < min_delta_t or dt > max_delta_t:
                continue

            key = (f1, f2, dt)
            keys.append(key)

    return keys


# ==========================
# 3. Fingerprint DB
# ==========================

class FingerprintDB:
    """
    여러 템플릿을 fingerprint key 집합으로 저장하는 DB.
    score(label) = |Q ∩ T_label| / sqrt(|Q| * |T_label|)
    """

    def __init__(self, sr: int = 16000, max_template_sec: float = 3.0):
        self.sr = sr
        self.max_template_sec = max_template_sec
        self.template_keys: Dict[str, set] = {}

    def add_template(self, label: str, audio_path: str):
        """템플릿 wav를 fingerprint key 집합으로 변환하여 저장."""
        y = load_mono(audio_path, sr=self.sr)
        y = crop_loudest_segment(y, sr=self.sr, max_sec=self.max_template_sec)

        keys = build_fingerprint_keys(y, sr=self.sr)
        key_set = set(keys)

        if label in self.template_keys:
            self.template_keys[label] |= key_set
        else:
            self.template_keys[label] = key_set

    def match_audio(self, y: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
        """
        입력 파형 y에 대해 fingerprint key 집합을 만들고
        각 템플릿과의 정규화된 교집합 비율을 score로 사용.
        """
        keys_q = build_fingerprint_keys(y, sr=self.sr)
        if len(keys_q) == 0:
            return None, 0.0, {}
        set_q = set(keys_q)
        len_q = float(len(set_q))

        scores: Dict[str, float] = {}
        for label, key_set in self.template_keys.items():
            if not key_set:
                continue

            inter = len(set_q & key_set)
            len_t = float(len(key_set))

            if inter == 0:
                score = 0.0
            else:
                score = inter / float(np.sqrt(len_q * len_t))

            scores[label] = score

        if not scores:
            return None, 0.0, {}

        best_label, best_score = max(scores.items(), key=lambda kv: kv[1])
        return best_label, best_score, scores


# ==========================
# 4. CustomDetector 클래스
# ==========================

class CustomDetector:
    """
    커스텀 소리 탐지를 위한 싱글톤 클래스.
    templates 폴더의 wav 파일들을 fingerprint로 등록하고,
    입력 오디오와 매칭하여 탐지합니다.
    """
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, template_dir: str = None, threshold: float = 0.3, sr: int = 16000):
        # 이미 초기화된 경우 스킵
        if CustomDetector._initialized:
            return
        
        self.sr = sr
        self.threshold = threshold
        self.db = FingerprintDB(sr=sr)
        self.template_dir = template_dir
        self.labels: List[str] = []
        
        if template_dir and os.path.isdir(template_dir):
            self._load_templates(template_dir)
        
        CustomDetector._initialized = True

    def _load_templates(self, template_dir: str):
        """템플릿 폴더에서 모든 wav 파일 로드."""
        for fname in sorted(os.listdir(template_dir)):
            if not fname.lower().endswith('.wav'):
                continue
            label = os.path.splitext(fname)[0]
            path = os.path.join(template_dir, fname)
            if os.path.exists(path):
                self.db.add_template(label, path)
                self.labels.append(label)

    def detect(self, audio: np.ndarray, sr: int = None) -> Tuple[Optional[str], float]:
        """
        오디오에서 커스텀 소리 탐지.
        
        Args:
            audio: 오디오 신호 (numpy array)
            sr: 샘플링 레이트 (None이면 리샘플링 안함)
        
        Returns:
            (label, score): 탐지된 라벨과 점수. 탐지 실패시 (None, 0.0)
        """
        if not self.labels:
            return None, 0.0
        
        # 샘플링 레이트가 다르면 리샘플링
        if sr is not None and sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        
        label, score, _ = self.db.match_audio(audio)
        
        if score >= self.threshold:
            return label, score
        return None, 0.0

    def has_templates(self) -> bool:
        """등록된 템플릿이 있는지 확인."""
        return len(self.labels) > 0
    
    @classmethod
    def reset(cls):
        """싱글톤 인스턴스 리셋 (테스트용)."""
        cls._instance = None
        cls._initialized = False


# ==========================
# 5. 편의 함수
# ==========================

_detector: Optional[CustomDetector] = None


def get_custom_detector(template_dir: str = None, threshold: float = 0.3) -> CustomDetector:
    """CustomDetector 싱글톤 인스턴스 반환."""
    global _detector
    if _detector is None:
        _detector = CustomDetector(template_dir=template_dir, threshold=threshold)
    return _detector


def try_custom_detection(audio: np.ndarray, sr: int, template_dir: str = None) -> Tuple[Optional[str], float]:
    """
    커스텀 소리 탐지 시도.
    
    Args:
        audio: 오디오 신호 (numpy array)
        sr: 샘플링 레이트
        template_dir: 템플릿 폴더 경로 (None이면 config에서 가져옴)
    
    Returns:
        (label, score): 탐지된 라벨과 점수. 탐지 실패시 (None, 0.0)
    """
    if template_dir is None:
        # config에서 가져오기
        try:
            from config import CUSTOM_TEMPLATE_DIR, CUSTOM_DETECTION_THRESHOLD
            template_dir = CUSTOM_TEMPLATE_DIR
            threshold = CUSTOM_DETECTION_THRESHOLD
        except ImportError:
            # config에 설정이 없으면 기본값 사용
            import os
            template_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "templates"
            )
            threshold = 0.3
    else:
        threshold = 0.3
    
    detector = get_custom_detector(template_dir=template_dir, threshold=threshold)
    return detector.detect(audio, sr=sr)
