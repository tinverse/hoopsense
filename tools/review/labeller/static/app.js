const player = document.getElementById('player');
const clipList = document.getElementById('clip-list');
const landmarkList = document.getElementById('landmark-list');
const calStats = document.getElementById('calibration-stats');
const currentDomain = document.getElementById('current-domain');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const explainerToggle = document.getElementById('explainer-toggle');
const explainerPanel = document.getElementById('explainer-panel');
const explainerTitle = document.getElementById('explainer-title');
const explainerDescription = document.getElementById('explainer-description');
const explainerStatus = document.getElementById('explainer-status');
const feedbackTrackList = document.getElementById('feedback-track-list');
const feedbackNote = document.getElementById('feedback-note');
const feedbackStatus = document.getElementById('feedback-status');

let clips = [];
let activeClip = null;
let calibrationPoints = [];
let isCalibrating = false;
let perceptionData = null;
let perceptionFrames = [];
let lastFeedbackFrameIdx = null;
let perceptionVisible = false;

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
        calibrationPoints = []; // Only reset on new clip
        loadPerceptionOverlay(activeClip);
        resetUI();
    }
});

function getVideoContentBox() {
    const rect = player.getBoundingClientRect();
    const videoRatio = player.videoWidth / player.videoHeight;
    const elementRatio = rect.width / rect.height;

    let contentWidth, contentHeight, offsetX, offsetY;

    if (elementRatio > videoRatio) {
        // Pillarboxed (bars on sides)
        contentHeight = rect.height;
        contentWidth = contentHeight * videoRatio;
        offsetX = (rect.width - contentWidth) / 2;
        offsetY = 0;
    } else {
        // Letterboxed (bars on top/bottom)
        contentWidth = rect.width;
        contentHeight = contentWidth / videoRatio;
        offsetX = 0;
        offsetY = (rect.height - contentHeight) / 2;
    }

    return {
        width: contentWidth,
        height: contentHeight,
        left: rect.left + offsetX,
        top: rect.top + offsetY,
        ratio: player.videoWidth / contentWidth
    };
}

function syncCanvasSize() {
    if (!(player.videoWidth && player.videoHeight)) {
        return;
    }
    const box = getVideoContentBox();
    
    // Position canvas EXACTLY over the video content using relative offsets
    // player.offsetLeft is relative to the #video-container
    overlay.style.width = box.width + 'px';
    overlay.style.height = box.height + 'px';
    overlay.style.left = (player.offsetLeft + (player.offsetWidth - box.width) / 2) + 'px';
    overlay.style.top = (player.offsetTop + (player.offsetHeight - box.height) / 2) + 'px';
    
    overlay.width = player.videoWidth;
    overlay.height = player.videoHeight;
}

window.addEventListener('resize', syncCanvasSize);

function resetUI() {
    isCalibrating = true;
    calStats.textContent = `Points: ${calibrationPoints.length}`;
    document.getElementById('calibration-hint').style.display = 'block';
    
    player.onloadedmetadata = () => {
        syncCanvasSize();
        redrawOverlay();
    };
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    redrawOverlay();
}

document.getElementById('btn-reset-cal').addEventListener('click', () => {
    calibrationPoints = [];
    resetUI();
});

document.getElementById('btn-run-cal').addEventListener('click', () => {
    if (calibrationPoints.length < 4) {
        alert("Need at least 4 points across the clip for calibration.");
        return;
    }
    finishCalibration();
});

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

document.getElementById('btn-id-yes').addEventListener('click', () => saveLabel('correct', 'id_verification'));
document.getElementById('btn-id-no').addEventListener('click', () => saveLabel('switched', 'id_verification'));
document.getElementById('btn-shot').addEventListener('click', () => saveLabel('jump_shot'));
document.getElementById('btn-pass').addEventListener('click', () => saveLabel('pass'));
document.getElementById('btn-dribble').addEventListener('click', () => saveLabel('dribble'));
document.getElementById('btn-idle').addEventListener('click', () => saveLabel('idle'));

function isTypingInFeedbackNote() {
    return document.activeElement === feedbackNote;
}

// 4. Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
    if (!activeClip) return;
    if (isTypingInFeedbackNote()) return;

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
            if (player.paused) player.play(); else player.pause();
            break;
        case 'ArrowRight':
            player.pause();
            player.currentTime += 1/30;
            break;
        case 'ArrowLeft':
            player.pause();
            player.currentTime -= 1/30;
            break;
        case 'r':
        case 'R':
            player.pause();
            player.currentTime = 0;
            break;
    }
});

// 5. Calibration Logic
player.addEventListener('mousedown', (e) => {
    if (!isCalibrating) return;

    const box = getVideoContentBox();
    
    // Calculate click relative to the actual video content start
    const clickX = e.clientX - box.left;
    const clickY = e.clientY - box.top;
    
    // Scale to raw video pixels
    const x = clickX * box.ratio;
    const y = clickY * box.ratio;
    
    const landmark_id = landmarkList.value;
    const t_ms = Math.floor(player.currentTime * 1000);

    if (clickX >= 0 && clickX <= box.width && clickY >= 0 && clickY <= box.height) {
        calibrationPoints.push({x, y, landmark_id, t_ms});
        drawPoint(x, y, landmark_id);
        calStats.textContent = `Points: ${calibrationPoints.length}`;
    }
});

function drawPoint(x, y, label) {
    ctx.fillStyle = "#ffaa00";
    ctx.beginPath();
    ctx.arc(x, y, 10, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();
    
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 16px monospace";
    ctx.fillText(label, x + 15, y + 5);
}

const COCO_EDGES = [
    [5, 7], [7, 9],
    [6, 8], [8, 10],
    [5, 6],
    [5, 11], [6, 12],
    [11, 12],
    [11, 13], [13, 15],
    [12, 14], [14, 16]
];

function getCurrentPerceptionFrame() {
    if (!perceptionVisible || !perceptionData || !perceptionData.enabled) return null;
    if (!perceptionFrames.length) return null;
    const targetMs = player.currentTime * 1000;
    let bestFrame = null;
    let bestDelta = Number.POSITIVE_INFINITY;
    for (const frame of perceptionFrames) {
        const delta = Math.abs(frame.t_ms - targetMs);
        if (delta < bestDelta) {
            bestDelta = delta;
            bestFrame = frame;
        }
    }
    return bestFrame;
}

function drawSkeleton(detection) {
    const keypoints = detection.keypoints_xy || [];
    const confidences = detection.keypoints_conf || [];

    ctx.save();
    ctx.strokeStyle = "#24ffe0";
    ctx.lineWidth = 3;
    COCO_EDGES.forEach(([a, b]) => {
        const ptA = keypoints[a];
        const ptB = keypoints[b];
        if (!ptA || !ptB) return;
        const confA = confidences[a] ?? 1.0;
        const confB = confidences[b] ?? 1.0;
        if (confA < 0.15 || confB < 0.15) return;
        ctx.beginPath();
        ctx.moveTo(ptA[0], ptA[1]);
        ctx.lineTo(ptB[0], ptB[1]);
        ctx.stroke();
    });

    keypoints.forEach((point, idx) => {
        const confidence = confidences[idx] ?? 1.0;
        if (!point || confidence < 0.15) return;
        ctx.fillStyle = idx < 5 ? "#ffd166" : "#24ffe0";
        ctx.beginPath();
        ctx.arc(point[0], point[1], 4, 0, Math.PI * 2);
        ctx.fill();
    });
    ctx.restore();
}

function drawDetectionOverlay(detection) {
    const [x1, y1, x2, y2] = detection.bbox_xyxy;
    const trackLabel = detection.track_id !== null && detection.track_id !== undefined
        ? `P${detection.track_id}`
        : detection.class_name;
    const uniformLabel = detection.uniform_bucket && detection.uniform_bucket !== 'unknown'
        ? ` ${detection.uniform_bucket.toUpperCase()}`
        : '';
    const confLabel = `${Math.round((detection.confidence || 0) * 100)}%`;
    const activeScore = detection.active_player_score !== undefined && detection.active_player_score !== null
        ? ` A${Math.round(detection.active_player_score * 100)}`
        : '';
    const repairLabel = detection.synthesized ? ' SYN' : '';
    const candidateColor = detection.active_player_candidate ? "#35f28b" : "#ff8f70";

    ctx.save();
    ctx.strokeStyle = detection.synthesized ? "#ffd166" : "#3ed8ff";
    ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${trackLabel}${uniformLabel} ${confLabel}${activeScore}${repairLabel}`;
    const labelWidth = Math.max(120, label.length * 10 + 20);
    const labelHeight = 26;
    const labelX = Math.max(10, Math.min(overlay.width - labelWidth - 10, x1));
    const labelY = Math.max(10, y1 - labelHeight - 6);

    ctx.fillStyle = "rgba(15, 39, 48, 0.92)";
    ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
    ctx.strokeStyle = candidateColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(labelX, labelY, labelWidth, labelHeight);
    ctx.fillStyle = detection.synthesized ? "#ffe3a3" : "#cfffff";
    ctx.font = "bold 16px monospace";
    ctx.fillText(label, labelX + 10, labelY + 17);
    ctx.restore();

    if (detection.keypoints_xy) {
        drawSkeleton(detection);
    }
}

function redrawOverlay() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    calibrationPoints.forEach(p => {
        if (Math.abs(p.t_ms - player.currentTime * 1000) < 100) {
            drawPoint(p.x, p.y, p.landmark_id);
        }
    });

    const frame = getCurrentPerceptionFrame();
    if (frame) {
        frame.detections.forEach(drawDetectionOverlay);
        explainerPanel.style.display = 'block';
        explainerTitle.textContent = perceptionData.title || 'Layer 1 Perception Overlay';
        const firstDetection = frame.detections.find(d => d.court_xy);
        const activeCount = frame.detections.filter(d => d.active_player_candidate).length;
        const synthCount = frame.detections.filter(d => d.synthesized).length;
        const courtText = firstDetection
            ? ` First court point: (${firstDetection.court_xy[0].toFixed(1)}, ${firstDetection.court_xy[1].toFixed(1)}).`
            : '';
        explainerDescription.textContent =
            `Real Ultralytics detections and pose keypoints for frame ${frame.frame_idx}. ` +
            `Green labels mark active-player candidates, orange labels mark likely sidelines/spectators, and yellow boxes are repaired track gaps. ` +
            `Cyan skeletons show keypoints.` +
            (frame.calibrated ? ` Calibration is active for this frame.${courtText}` : ' No calibration is active for this frame.');
        explainerStatus.textContent =
            `${frame.detections.length} detections // ${activeCount} active candidates // ${synthCount} repaired // ${Math.round(frame.t_ms / 10) / 100}s // ${perceptionData.model.name} // ${frame.calibrated ? 'calibrated' : 'raw'}`;
    } else if (perceptionVisible && perceptionData && perceptionData.enabled) {
        explainerPanel.style.display = 'block';
        explainerTitle.textContent = perceptionData.title || 'Layer 1 Perception Overlay';
        explainerDescription.textContent = 'Play or step through the clip to inspect the current frame’s tracked detections and keypoints.';
        explainerStatus.textContent = 'No frame-aligned detections at this timestamp';
    }
}

function populateFeedbackTrackList() {
    const frame = getCurrentPerceptionFrame();
    const nextFrameIdx = frame ? frame.frame_idx : null;
    if (nextFrameIdx === lastFeedbackFrameIdx) {
        return;
    }
    lastFeedbackFrameIdx = nextFrameIdx;

    feedbackTrackList.innerHTML = '';
    const base = document.createElement('option');
    base.value = '';
    base.textContent = 'Frame-level issue / no specific track';
    feedbackTrackList.appendChild(base);
    if (!frame) return;
    frame.detections.forEach((detection, index) => {
        const opt = document.createElement('option');
        const trackId = detection.track_id !== null && detection.track_id !== undefined
            ? String(detection.track_id)
            : `det-${index}`;
        opt.value = trackId;
        const uniformLabel = detection.uniform_bucket ? ` // ${detection.uniform_bucket}` : '';
        const activeLabel = detection.active_player_score !== undefined && detection.active_player_score !== null
            ? ` // active ${Math.round(detection.active_player_score * 100)}`
            : '';
        const synthLabel = detection.synthesized ? ' // repaired' : '';
        opt.textContent = `Track ${trackId}${uniformLabel} // ${Math.round((detection.confidence || 0) * 100)}%${activeLabel}${synthLabel}`;
        feedbackTrackList.appendChild(opt);
    });
}

function savePerceptionFeedback(issueType) {
    if (!(activeClip && perceptionData && perceptionData.enabled)) return;
    const frame = getCurrentPerceptionFrame();
    if (!frame) {
        feedbackStatus.textContent = 'No active frame artifact at this timestamp.';
        return;
    }

    const payload = {
        clip_id: activeClip.id,
        domain: activeClip.domain,
        frame_idx: frame.frame_idx,
        t_ms: frame.t_ms,
        track_id: feedbackTrackList.value || null,
        issue_type: issueType,
        note: feedbackNote.value.trim(),
        timestamp: new Date().toISOString()
    };

    fetch('/api/perception_feedback', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
    })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                feedbackStatus.textContent = `Saved ${issueType} at frame ${frame.frame_idx}${payload.track_id ? ` for ${payload.track_id}` : ''}.`;
                feedbackNote.value = '';
            } else {
                feedbackStatus.textContent = `Save failed: ${JSON.stringify(data)}`;
            }
        });
}

function updateExplainerButton() {
    const available = !!(perceptionData && perceptionData.enabled);
    explainerToggle.disabled = !available;
    explainerToggle.style.opacity = available ? "1" : "0.45";
    if (!available) {
        explainerToggle.textContent = 'NO PERCEPTION OVERLAY FOR THIS CLIP';
        return;
    }
    explainerToggle.textContent = perceptionVisible ? 'HIDE PERCEPTION OVERLAY' : 'SHOW PERCEPTION OVERLAY';
}

function loadPerceptionOverlay(clip) {
    perceptionVisible = false;
    perceptionFrames = [];
    lastFeedbackFrameIdx = null;
    fetch(`/api/perception/${clip.id}`)
        .then(res => res.json())
        .then(data => {
            perceptionData = data;
            const frames = Array.isArray(data.frames) ? data.frames : [];
            perceptionFrames = frames;
            explainerPanel.style.display = data.enabled ? 'block' : 'none';
            if (data.enabled) {
                explainerTitle.textContent = 'Layer 1 Perception Overlay';
                explainerDescription.textContent =
                    `Artifact ready for ${clip.id}. Toggle the overlay and step through the clip to inspect real detections, pose, and calibrated court coordinates when available.`;
                explainerStatus.textContent = data.calibration && data.calibration.enabled ? 'Ready // calibration available' : 'Ready // raw perception only';
                feedbackStatus.textContent = 'Select a frame and optionally a track, then save structured feedback.';
            } else {
                feedbackStatus.textContent = 'No perception artifact for this clip yet.';
            }
            updateExplainerButton();
            populateFeedbackTrackList();
            redrawOverlay();
        })
        .catch((error) => {
            perceptionData = null;
            perceptionFrames = [];
            explainerPanel.style.display = 'none';
            feedbackStatus.textContent = `Failed to load perception artifact: ${error.message}`;
            updateExplainerButton();
        });
}

explainerToggle.addEventListener('click', () => {
    if (!(perceptionData && perceptionData.enabled)) return;
    perceptionVisible = !perceptionVisible;
    updateExplainerButton();
    redrawOverlay();
});

player.addEventListener('timeupdate', redrawOverlay);
player.addEventListener('timeupdate', populateFeedbackTrackList);
player.addEventListener('seeked', redrawOverlay);
player.addEventListener('seeked', populateFeedbackTrackList);

document.getElementById('feedback-fp').addEventListener('click', () => savePerceptionFeedback('false_positive'));
document.getElementById('feedback-fn').addEventListener('click', () => savePerceptionFeedback('false_negative'));
document.getElementById('feedback-merge').addEventListener('click', () => savePerceptionFeedback('merge_error'));
document.getElementById('feedback-track').addEventListener('click', () => savePerceptionFeedback('track_error'));
document.getElementById('feedback-pose').addEventListener('click', () => savePerceptionFeedback('pose_error'));

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
          console.log("Panning Calibration Complete. Frames:", data.frames_calibrated);
          alert(`Panning Calibration Saved! Tracked ${data.frames_calibrated} frames.`);
      });
    setTimeout(() => ctx.clearRect(0, 0, overlay.width, overlay.height), 2000);
}
