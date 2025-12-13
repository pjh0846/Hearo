// ============================================================
// 전역 상태
// ============================================================

let timelineData = [];
let currentFrameIndex = 0;
let isPlaying = false;
let playInterval = null;
let audioElement = null; // 오디오 재생용
const AUDIO_OFFSET = 5; // 오디오 시작 오프셋 (5초 윈도우 길이)


// ============================================================
// 파일 업로드 및 분석
// ============================================================

document.getElementById('fileInput').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        analyzeFile(file);
    }
});

function analyzeFile(file) {
    document.getElementById('loading').classList.add('show');
    document.getElementById('resultSection').classList.remove('show');
    document.getElementById('noSound').classList.remove('show');

    // 오디오 파일 로드
    if (audioElement) {
        audioElement.pause();
        audioElement = null;
    }
    audioElement = new Audio(URL.createObjectURL(file));

    const formData = new FormData();
    formData.append('file', file);

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error); });
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('loading').classList.remove('show');
            alert('분석 오류: ' + error.message);
        });
}

// ============================================================
// 결과 표시
// ============================================================

function displayResults(data) {
    document.getElementById('loading').classList.remove('show');

    if (!data.detected || data.detections.length === 0) {
        document.getElementById('noSound').classList.add('show');
        return;
    }

    document.getElementById('resultSection').classList.add('show');

    // 타임라인 데이터 저장
    timelineData = data.timeline || [];

    if (timelineData.length > 0) {
        // 타임라인 컨트롤 초기화
        initTimeline();
        // 첫 번째 프레임 표시
        updateFrame(0);
        // 자동 재생 시작
        playTimeline();
    } else {
        // 타임라인이 없으면 기존 방식 (전체 요약)
        document.getElementById('timelineControls').style.display = 'none';
        displayStaticResults(data.detections);
    }
}

function initTimeline() {
    const controls = document.getElementById('timelineControls');
    controls.style.display = 'flex';

    const slider = document.getElementById('timelineSlider');
    slider.max = timelineData.length - 1;
    slider.value = 0;

    const totalTime = timelineData[timelineData.length - 1].timestamp;
    document.getElementById('totalTime').textContent = totalTime.toFixed(1) + 's';

    // 이벤트 리스너
    document.getElementById('playButton').addEventListener('click', togglePlay);
    slider.addEventListener('input', function () {
        pauseTimeline();
        const index = parseInt(this.value);
        updateFrame(index);
        // 오디오 시간도 동기화 (타임라인 timestamp + 오프셋)
        if (audioElement) {
            audioElement.currentTime = timelineData[index].timestamp + AUDIO_OFFSET;
        }
    });
}

function togglePlay() {
    if (isPlaying) {
        pauseTimeline();
    } else {
        playTimeline();
    }
}

function playTimeline() {
    isPlaying = true;
    document.getElementById('playButton').textContent = '⏸️';

    // 오디오 재생
    if (audioElement) {
        // 타임라인 timestamp + 오프셋으로 오디오 시작 위치 계산
        const startTime = timelineData[currentFrameIndex].timestamp + AUDIO_OFFSET;

        // 오디오가 로드되었는지 확인
        if (audioElement.readyState >= 2) {
            audioElement.currentTime = startTime;
            audioElement.play().catch(err => console.error('Audio play error:', err));
        } else {
            // 로딩 대기 후 재생
            audioElement.addEventListener('canplay', function onCanPlay() {
                audioElement.currentTime = timelineData[currentFrameIndex].timestamp + AUDIO_OFFSET;
                audioElement.play().catch(err => console.error('Audio play error:', err));
                audioElement.removeEventListener('canplay', onCanPlay);
            });
        }

        // 오디오 시간에 따라 프레임 업데이트
        audioElement.addEventListener('timeupdate', syncFrameWithAudio);
        audioElement.addEventListener('ended', onAudioEnded);
    } else {
        // 오디오 없으면 타이머로 재생
        playInterval = setInterval(() => {
            currentFrameIndex++;
            if (currentFrameIndex >= timelineData.length) {
                currentFrameIndex = 0; // 루프
            }
            updateFrame(currentFrameIndex);
        }, 500); // 0.5초마다 프레임 업데이트
    }
}

function pauseTimeline() {
    isPlaying = false;
    document.getElementById('playButton').textContent = '▶️';

    // 오디오 일시정지
    if (audioElement) {
        audioElement.pause();
        audioElement.removeEventListener('timeupdate', syncFrameWithAudio);
        audioElement.removeEventListener('ended', onAudioEnded);
    }

    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
    }
}

function onAudioEnded() {
    pauseTimeline();
    updateFrame(0);
    if (audioElement) {
        audioElement.currentTime = 0;
    }
}

function syncFrameWithAudio() {
    if (!isPlaying || !audioElement) return;

    // 오디오 시간에서 오프셋을 빼서 타임라인 시간으로 변환
    const currentTime = audioElement.currentTime - AUDIO_OFFSET;

    // 이진 탐색으로 현재 시간에 맞는 프레임 찾기 (더 정확하고 빠름)
    let closestIndex = findClosestFrameIndex(currentTime);

    if (closestIndex !== currentFrameIndex) {
        updateFrame(closestIndex);
    }
}

function findClosestFrameIndex(targetTime) {
    // 이진 탐색으로 가장 가까운 프레임 찾기
    let left = 0;
    let right = timelineData.length - 1;

    while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (timelineData[mid].timestamp < targetTime) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }

    // left와 left-1 중 더 가까운 것 선택
    if (left > 0) {
        const diffLeft = Math.abs(timelineData[left].timestamp - targetTime);
        const diffPrev = Math.abs(timelineData[left - 1].timestamp - targetTime);
        return diffPrev < diffLeft ? left - 1 : left;
    }

    return left;
}

function updateFrame(index) {
    currentFrameIndex = index;
    const frame = timelineData[index];

    // 슬라이더 업데이트
    document.getElementById('timelineSlider').value = index;
    document.getElementById('currentTime').textContent = frame.timestamp.toFixed(1) + 's';

    // 레이더 및 목록 업데이트
    displayStaticResults(frame.detections);
}

// ============================================================
// 정적 결과 표시 (레이더 + 목록)
// ============================================================

function displayStaticResults(detections) {
    // 레이더에 마커 표시
    const markersContainer = document.getElementById('soundMarkers');
    markersContainer.innerHTML = '';

    // 감지 목록 표시
    const listContainer = document.getElementById('detectionList');
    listContainer.innerHTML = '';

    detections.forEach((det, index) => {
        // 레이더 마커 추가
        const marker = document.createElement('div');
        marker.className = 'sound-marker' + (det.urgency > 0.7 ? ' urgent' : '');
        marker.textContent = det.icon;

        let x, y;
        if (det.direction === 'center') {
            // 커스텀 소리: 레이더 중앙(사용자 위치)에 표시
            marker.title = det.class_kr;
            x = 150;
            y = 150;
            marker.classList.add('center-marker');
        } else {
            // SELD 소리: 방향에 따라 표시
            marker.title = `${det.class_kr} (${det.azimuth}°)`;
            const radius = 100;
            const angleRad = (det.azimuth - 90) * Math.PI / 180;
            x = 150 + radius * Math.cos(angleRad);
            y = 150 + radius * Math.sin(angleRad);
        }

        marker.style.left = x + 'px';
        marker.style.top = y + 'px';
        markersContainer.appendChild(marker);

        // 목록 아이템 추가
        const item = document.createElement('div');
        item.className = 'detection-item' + (det.urgency > 0.7 ? ' urgent' : '');

        // 방향 정보 표시 (center면 생략)
        const directionText = det.direction === 'center'
            ? ''
            : `${det.direction_kr} (${det.azimuth > 0 ? '+' : ''}${det.azimuth}°)`;

        item.innerHTML = `
            <div class="detection-icon">${det.icon}</div>
            <div class="detection-info">
                <div class="detection-name">${det.class_kr}</div>
                ${directionText ? `<div class="detection-direction">${directionText}</div>` : ''}
            </div>
            ${det.arrow ? `<div class="detection-arrow${det.urgency > 0.7 ? ' urgent' : ''}">${det.arrow}</div>` : ''}
        `;
        listContainer.appendChild(item);
    });
}
