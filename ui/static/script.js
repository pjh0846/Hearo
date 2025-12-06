const SOUND_ICONS = {
    'siren': '🚨',
    'car_horn': '🚗',
    'glass_breaking': '💥',
    'train': '🚂',
    'engine': '🔧',
    'door_wood_knock': '🚪',
    'crying_baby': '👶',
    'dog': '🐕',
    'footsteps': '👣',
    'clock_alarm': '⏰',
    'other': '🔊'
};

document.getElementById('fileInput').addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        analyzeFile(file);
    }
});

function analyzeFile(file) {
    document.getElementById('loading').classList.add('show');
    document.getElementById('resultCard').classList.remove('show');

    const formData = new FormData();
    formData.append('file', file);

    fetch('/analyze', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('분석 실패');
            }
            return response.json();
        })
        .then(data => {
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('loading').classList.remove('show');
            alert('파일 분석 중 오류가 발생했습니다: ' + error.message);
        });
}

function displayResults(data) {
    document.getElementById('loading').classList.remove('show');
    document.getElementById('resultCard').classList.add('show');

    // 아이콘 + 세기에 따른 크기
    const icon = SOUND_ICONS[data.sound_type.name_en] || '🔊';
    const iconEl = document.getElementById('soundIcon');
    iconEl.textContent = icon;

    // 세기에 따라 아이콘 크기 설정
    iconEl.className = `sound-icon intensity-${getLevelClass(data.intensity.level)}`;

    // 긴급도가 0.6 초과면 진동 효과 추가
    if (data.urgency.value > 0.6) {
        iconEl.classList.add('shake');
    } else {
        iconEl.classList.remove('shake');
    }

    // 파동 생성 (제거됨)
    const wavesContainer = document.getElementById('soundWaves');
    wavesContainer.innerHTML = '';

    // 소리 이름 표시
    const soundNameEl = document.getElementById('soundName');
    soundNameEl.textContent = data.sound_type.name_kr;
    soundNameEl.style.color = '#333'; // 검은색으로 통일
}

function getLevelClass(level) {
    const map = {
        '높음': 'high', '상': 'high',
        '보통': 'medium', '중': 'medium',
        '낮음': 'low', '하': 'low'
    };
    return map[level] || 'medium';
}

function getUrgencyColor(level) {
    const colors = {
        '높음': '#ff4b2b',
        '보통': '#ffd200',
        '낮음': '#a8e063'
    };
    return colors[level] || '#ffd200';
}

