const player = document.getElementById('player');
const clipList = document.getElementById('clip-list');
const landmarkList = document.getElementById('landmark-list');
const calStats = document.getElementById('calibration-stats');
const currentDomain = document.getElementById('current-domain');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');

let clips = [];
let activeClip = null;
let calibrationPoints = [];
let isCalibrating = false;

// 1. Fetch clips on load
fetch('/api/clips')
    .then(res => res.json())
    .then(data => {
        clips = data;
        data.forEach(clip => {
            const opt = document.createElement('option');
            opt.value = clip.path;
            opt.textContent = `[${clip.domain}] ${clip.id}`;
            clipList.appendChild(opt);
        });
    });

// 2. Handle Clip Selection
clipList.addEventListener('change', (e) => {
    const path = e.target.value;
    activeClip = clips.find(c => c.path === path);
    if (activeClip) {
        player.src = `/api/video/${activeClip.path}`;
        currentDomain.textContent = activeClip.domain.toUpperCase();
        resetUI();
    }
});

function resetUI() {
    calibrationPoints = [];
    isCalibrating = true;
    calStats.textContent = "Points: 0/4";
    document.getElementById('calibration-hint').style.display = 'block';
    player.onloadedmetadata = () => {
        overlay.width = player.videoWidth;
        overlay.height = player.videoHeight;
    };
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}

document.getElementById('btn-reset-cal').addEventListener('click', resetUI);

// 3. Labelling Actions
function saveLabel(label, type="action") {
    if (!activeClip) return;
    
    const data = {
        clip_id: activeClip.id,
        domain: activeClip.domain,
        t_ms: Math.floor(player.currentTime * 1000),
        type: type,
        label: label,
        timestamp: new Date().toISOString()
    };

    fetch('/api/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    }).then(() => {
        console.log("Saved:", label);
    });
}

// 4. Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
    if (!activeClip) return;

    switch(e.key) {
        case ' ': 
            e.preventDefault();
            saveLabel('correct', 'id_verification');
            break;
        case 'x':
        case 'X':
            saveLabel('switched', 'id_verification');
            break;
        case '1': saveLabel('jump_shot'); break;
        case '2': saveLabel('pass'); break;
        case '3': saveLabel('dribble'); break;
        case '4': saveLabel('idle'); break;
        case 'p':
        case 'P':
            player.paused ? player.play() : player.pause();
            break;
        case 'ArrowRight':
            player.pause();
            player.currentTime += 1/30;
            break;
        case 'ArrowLeft':
            player.pause();
            player.currentTime -= 1/30;
            break;
    }
});

// 5. Calibration Logic (Clicking landmarks)
player.addEventListener('mousedown', (e) => {
    if (!isCalibrating) return;

    const rect = player.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (player.videoWidth / rect.width);
    const y = (e.clientY - rect.top) * (player.videoHeight / rect.height);
    const landmark_id = landmarkList.value;

    calibrationPoints.push({x, y, landmark_id});
    drawPoint(x, y, landmark_id);
    calStats.textContent = `Points: ${calibrationPoints.length}/4`;

    if (calibrationPoints.length === 4) {
        finishCalibration();
    }
});

function drawPoint(x, y, label) {
    ctx.fillStyle = "#ffaa00";
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.font = "12px monospace";
    ctx.fillText(label, x + 8, y + 4);
}

function finishCalibration() {
    isCalibrating = false;
    document.getElementById('calibration-hint').style.display = 'none';
    
    fetch('/api/calibrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            id: activeClip.id,
            path: activeClip.path,
            t_ms: Math.floor(player.currentTime * 1000),
            points: calibrationPoints
        })
    }).then(res => res.json())
      .then(data => {
          console.log("Panning Calibration Complete. Frames:", data.frames_tracked);
          alert(`Panning Calibration Saved! Tracked ${data.frames_tracked} frames.`);
      });
    ctx.clearRect(0, 0, overlay.width, overlay.height);
}
