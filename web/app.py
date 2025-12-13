"""
SELD Web UI Backend
SELD 모델 전용 Flask 서버 - 방향 감지 + 커스텀 소리 감지 지원
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import numpy as np

# ============================================================
# 경로 설정 및 의존성 로드
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BACKEND_DIR = os.path.join(PROJECT_ROOT, 'backend')
SELD_DIR = os.path.join(BACKEND_DIR, 'seld')
CUSTOM_DIR = os.path.join(BACKEND_DIR, 'custom')
sys.path.insert(0, SELD_DIR)
sys.path.insert(0, CUSTOM_DIR)
sys.path.insert(0, BACKEND_DIR)

from inference import RealtimeSELD
import inference_utils as utils
from custom_detector import CustomDetector, get_custom_detector
from config import CUSTOM_TEMPLATE_DIR, CUSTOM_DETECTION_THRESHOLD

# ============================================================
# 상수 정의
# ============================================================

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(WEB_DIR, 'uploads_seld')

# 클래스별 아이콘 및 한글 매핑
CLASS_INFO = {
    'alarm': {'icon': '⏰', 'kr': '알람'},
    'baby': {'icon': '👶', 'kr': '아기 울음'},
    'crash': {'icon': '💥', 'kr': '충돌'},
    'dog': {'icon': '🐕', 'kr': '개 짖는 소리'},
    'engine': {'icon': '🚘', 'kr': '엔진'},
    'female_scream': {'icon': '😱', 'kr': '여성 비명'},
    'female_speech': {'icon': '👩', 'kr': '여성 말소리'},
    'fire': {'icon': '🔥', 'kr': '화재'},
    'footsteps': {'icon': '👣', 'kr': '발소리'},
    'knock': {'icon': '🚪', 'kr': '노크'},
    'male_scream': {'icon': '😨', 'kr': '남성 비명'},
    'male_speech': {'icon': '👨', 'kr': '남성 말소리'},
    'phone': {'icon': '📱', 'kr': '전화'},
    'piano': {'icon': '🎹', 'kr': '피아노'},
    'custom': {'icon': '🔔', 'kr': '커스텀'}
}

# 클래스별 긴급도 (0.0 ~ 1.0)
URGENCY_RULES = {
    'alarm': 0.85, 'baby': 0.75, 'crash': 0.95, 'dog': 0.60, 'engine': 0.40,
    'female_scream': 0.90, 'female_speech': 0.30, 'fire': 0.95, 'footsteps': 0.20,
    'knock': 0.30, 'male_scream': 0.90, 'male_speech': 0.30, 'phone': 0.70, 
    'piano': 0.20, 'custom': 0.70
}

# 커스텀 소리 라벨 매핑 (파일명 -> 한글 표시명)
CUSTOM_LABEL_MAP = {
    'doorbellH': '우리집 초인종',
    'doorbellA': '초인종 A',
    'doorbellB': '초인종 B',
    'doorbellC': '초인종 C',
    'doorbellD': '초인종 D',
    'doorbellE': '우리집 초인종',
    'doorbellF': '우리집 초인종',
    'doorbellG': '초인종 G',
    'doorbellI': '초인종 I',
    'doorbellJ': '초인종 J',
    'LG': 'LG 세탁기',
    'SAMSUNG': '삼성 세탁기',
    'microwave': '전자레인지'
}

# ============================================================
# 헬퍼 함수
# ============================================================

def get_direction_info(azimuth: float) -> dict:
    """방위각을 방향 정보로 변환 (8방향)"""
    # 8방향 분류: 22.5도씩 분할
    if -22.5 <= azimuth < 22.5:
        return {'direction': 'front', 'kr': '정면', 'arrow': '↑'}
    elif 22.5 <= azimuth < 67.5:
        return {'direction': 'front-right', 'kr': '오른쪽 앞', 'arrow': '↗'}
    elif 67.5 <= azimuth < 112.5:
        return {'direction': 'right', 'kr': '오른쪽', 'arrow': '→'}
    elif 112.5 <= azimuth < 157.5:
        return {'direction': 'back-right', 'kr': '오른쪽 뒤', 'arrow': '↘'}
    elif 157.5 <= azimuth <= 180 or -180 <= azimuth < -157.5:
        return {'direction': 'back', 'kr': '뒤쪽', 'arrow': '↓'}
    elif -157.5 <= azimuth < -112.5:
        return {'direction': 'back-left', 'kr': '왼쪽 뒤', 'arrow': '↙'}
    elif -112.5 <= azimuth < -67.5:
        return {'direction': 'left', 'kr': '왼쪽', 'arrow': '←'}
    else:  # -67.5 <= azimuth < -22.5
        return {'direction': 'front-left', 'kr': '왼쪽 앞', 'arrow': '↖'}



def build_custom_response(label: str, score: float, timestamp: float = 0.0) -> dict:
    """커스텀 소리 감지 결과 생성"""
    # 라벨 매핑 적용 (파일명 -> 한글 표시명)
    display_label = CUSTOM_LABEL_MAP.get(label, label)
    
    detection = {
        'class_en': f'custom:{label}',
        'class_kr': display_label,
        'icon': '🔔',
        'azimuth': None,
        'direction': 'center',
        'direction_kr': '',
        'arrow': '',
        'urgency': URGENCY_RULES.get('custom', 0.7),
        'is_custom': True,
        'custom_score': round(score, 3)
    }
    
    return {
        'detected': True,
        'detections': [detection],
        'timeline': [{
            'timestamp': round(timestamp, 1),
            'detections': [detection]
        }]
    }


def build_seld_detection(class_name: str, azimuth: float) -> dict:
    """SELD 감지 결과를 API 응답 형식으로 변환"""
    direction = get_direction_info(azimuth)
    info = CLASS_INFO.get(class_name, {'icon': '🔊', 'kr': class_name})
    
    return {
        'class_en': class_name,
        'class_kr': info['kr'],
        'icon': info['icon'],
        'azimuth': round(azimuth, 1),
        'direction': direction['direction'],
        'direction_kr': direction['kr'],
        'arrow': direction['arrow'],
        'urgency': URGENCY_RULES.get(class_name, 0.5)
    }


def run_seld_analysis(audio: np.ndarray, engine: RealtimeSELD) -> tuple:
    """SELD 모델로 전체 오디오 분석 - 타임라인 데이터 포함"""
    total_samples = audio.shape[1]
    slide_samples = int(engine.sr * 0.5)  # 0.5초 간격
    all_detections = {}  # 전체 요약용 (클래스별 마지막 감지)
    timeline_frames = []  # 프레임별 데이터
    
    pos = 0
    frame_index = 0
    while pos + engine.window_samples <= total_samples:
        window = audio[:, pos:pos + engine.window_samples]
        features = engine.extract_features(window)
        result = engine.infer(features)
        detections = engine.parse_results(result)
        
        # 타임라인 프레임 저장
        if detections:
            # 0초부터 시작하는 타임스탬프 (프론트엔드에서 오디오 오프셋 적용)
            timestamp = pos / engine.sr
            timeline_frames.append({
                'timestamp': round(timestamp, 1),
                'detections': detections.copy()
            })
        
        # 전체 요약용 (기존 로직 유지)
        for det in detections:
            all_detections[det['class']] = det
        
        pos += slide_samples
        frame_index += 1
    
    return all_detections, timeline_frames


# ============================================================
# Flask 앱 초기화
# ============================================================

app = Flask(__name__, static_folder='static')
CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 커스텀 소리 탐지기 초기화
CustomDetector.reset()
custom_detector = get_custom_detector(
    template_dir=CUSTOM_TEMPLATE_DIR, 
    threshold=CUSTOM_DETECTION_THRESHOLD
)
print(f"📁 커스텀 템플릿 로드 완료: {len(custom_detector.labels)}개 ({custom_detector.labels})")

# SELD 모델 로드
print("🔄 SELD 모델 로딩 중...")
seld_engine = RealtimeSELD(
    weights_dir=os.path.join(SELD_DIR, 'weights'),
    device='cuda',
    slide_interval_ms=200
)
print("✅ SELD 모델 로드 완료!")

# ============================================================
# 라우트 정의
# ============================================================

@app.route('/')
def index():
    """메인 페이지"""
    return send_from_directory('.', 'index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """SELD 분석 API"""
    # 파일 유효성 검사
    if 'file' not in request.files:
        return jsonify({'error': '파일이 없습니다'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
    
    # 파일 저장
    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filename)
    
    try:
        # 오디오 로드
        audio, _ = utils.load_audio(filename, seld_engine.sr)
        
        # 스테레오 확인
        if audio.shape[0] != 2:
            return jsonify({
                'error': f'스테레오 오디오가 필요합니다. 현재: {audio.shape[0]}채널'
            }), 400
        
        # 최소 길이 패딩
        total_samples = audio.shape[1]
        if total_samples < seld_engine.window_samples:
            pad_length = seld_engine.window_samples - total_samples
            audio = np.pad(audio, ((0, 0), (0, pad_length)), mode='constant')
        
        # 1순위: 커스텀 소리 탐지 (전체 오디오에서 한 번만)
        if custom_detector.has_templates():
            label, score = custom_detector.detect(audio[0], sr=seld_engine.sr)
            if label:
                return jsonify(build_custom_response(label, score, 0.0))
        
        # 2순위: SELD 분석
        all_detections, timeline_frames = run_seld_analysis(audio, seld_engine)
        
        # 결과 변환
        detection_list = [
            build_seld_detection(name, float(det['azimuth']))
            for name, det in all_detections.items()
        ]
        
        if not detection_list:
            return jsonify({
                'detected': False,
                'message': '감지된 소리가 없습니다',
                'detections': [],
                'timeline': []
            })
        
        # 긴급도 순 정렬
        detection_list.sort(key=lambda x: x['urgency'], reverse=True)
        
        # 타임라인 데이터 변환
        timeline_data = []
        for frame in timeline_frames:
            frame_detections = [
                build_seld_detection(det['class'], float(det['azimuth']))
                for det in frame['detections']
            ]
            frame_detections.sort(key=lambda x: x['urgency'], reverse=True)
            timeline_data.append({
                'timestamp': frame['timestamp'],
                'detections': frame_detections
            })
        
        return jsonify({
            'detected': True,
            'detections': detection_list,
            'timeline': timeline_data
        })
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
        
    finally:
        # 임시 파일 삭제
        try:
            os.remove(filename)
        except:
            pass


# ============================================================
# 메인 실행
# ============================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Hearo SELD Web UI 서버 시작")
    print("="*60)
    print("\n브라우저에서 접속: http://localhost:5000\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

