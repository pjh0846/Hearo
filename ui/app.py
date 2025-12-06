"""
Hearo Web UI Backend
Flask 서버로 프론트엔드와 연결
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys

# 프로젝트 경로 추가 (src 폴더를 루트로 인식하도록)
# ui/app.py -> .. -> src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from models.predict import predict_audio_features
from models.custom_detector import CustomDetector, get_custom_detector
from config import CUSTOM_TEMPLATE_DIR, CUSTOM_DETECTION_THRESHOLD

# 앱 시작 시 싱글톤 리셋 및 템플릿 새로 로드
CustomDetector.reset()
detector = get_custom_detector(template_dir=CUSTOM_TEMPLATE_DIR, threshold=CUSTOM_DETECTION_THRESHOLD)
print(f"📁 커스텀 템플릿 로드 완료: {len(detector.labels)}개 ({detector.labels})")

# Flask 앱 초기화
app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    """메인 페이지"""
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    오디오/비디오 파일 분석 API
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
        
        # 파일 저장
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        
        # 분석 실행
        features = predict_audio_features(filename)
        
        # 임시 파일 삭제
        try:
            os.remove(filename)
        except:
            pass
        
        return jsonify(features)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎧 Hearo Web UI 서버 시작")
    print("="*60)
    print("\n브라우저에서 접속: http://localhost:5000\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
