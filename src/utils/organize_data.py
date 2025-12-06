import os
import sys
import shutil
from tqdm import tqdm

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config import PROJECT_ROOT

# ESC-50 클래스 ID 매핑 (ESC-50 ID -> 프로젝트 클래스 이름)
ESC50_TO_PROJECT_CLASS = {
    42: 'siren',            # Siren
    43: 'car_horn',         # Car horn
    39: 'glass_breaking',   # Glass breaking
    45: 'train',            # Train
    44: 'engine',           # Engine
    30: 'door_wood_knock',  # Door knock
    20: 'crying_baby',      # Crying baby
    0: 'dog',               # Dog
    25: 'footsteps',        # Footsteps
    37: 'clock_alarm',      # Clock alarm
}

def organize_data():
    # 경로 설정
    raw_data_dir = os.path.join(PROJECT_ROOT, 'data', 'raw', 'ESC-50', 'audio')
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'classified')
    
    if not os.path.exists(raw_data_dir):
        print(f"❌ 데이터 디렉토리를 찾을 수 없습니다: {raw_data_dir}")
        return

    # 기존 분류 디렉토리 초기화
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # 파일 목록 가져오기
    files = [f for f in os.listdir(raw_data_dir) if f.endswith('.wav')]
    total_files = len(files)
    print(f"📊 총 {total_files}개의 파일을 분류합니다 (파일명 기반).")
    
    # 분류 시작
    success_count = 0
    error_count = 0
    
    for filename in tqdm(files, desc="분류 중"):
        file_path = os.path.join(raw_data_dir, filename)
        
        try:
            # 파일명 파싱: {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav
            # 예: 1-100032-A-0.wav -> target = 0
            parts = filename.split('-')
            target_id = int(parts[-1].split('.')[0])
            
            # 클래스 결정
            class_name = ESC50_TO_PROJECT_CLASS.get(target_id, 'other')
            
            # 대상 디렉토리 생성
            target_dir = os.path.join(output_dir, class_name)
            os.makedirs(target_dir, exist_ok=True)
            
            # 파일 복사
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(file_path, target_path)
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ 오류 발생 ({filename}): {e}")
            error_count += 1
            
    print("\n✅ 분류 완료!")
    print(f"   - 성공: {success_count}개")
    print(f"   - 실패: {error_count}개")
    print(f"   - 저장 위치: {output_dir}")

if __name__ == "__main__":
    organize_data()
