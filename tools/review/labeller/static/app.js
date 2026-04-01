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
const playPauseButton = document.getElementById('btn-play-pause');
const frameReadout = document.getElementById('frame-readout');

let clips = [];
let activeClip = null;
let calibrationPoints = [];
let landmarkSpecs = {};
let isCalibrating = false;
let perceptionData = null;
let perceptionFrames = [];
let lastFeedbackFrameIdx = null;
let perceptionVisible = false;
let playbackOverlayHandle = null;
const JERSEY_UI_MIN_CONFIDENCE = 0.9;
const JERSEY_UI_MIN_EVIDENCE = 3;

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

fetch('/api/landmarks')
    .then(res => res.json())
    .then(data => {
        landmarkSpecs = data || {};
        populateLandmarkList();
        updateCalibrationStats();
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

function clampVideoTime(seconds) {
    const duration = Number.isFinite(player.duration) ? player.duration : null;
    if (duration === null) return Math.max(0, seconds);
    return Math.min(Math.max(0, seconds), duration);
}

function syncPerceptionUI() {
    redrawOverlay();
    populateFeedbackTrackList();
    updateFrameReadout();
    updateTransportButton();
}

function updateFrameReadout() {
    if (!frameReadout) return;
    const frame = getCurrentPerceptionFrame();
    if (frame) {
        frameReadout.textContent = `Frame: ${frame.frame_idx} // Time: ${(frame.t_ms / 1000).toFixed(2)}s`;
        return;
    }
    frameReadout.textContent = `Frame: - // Time: ${player.currentTime.toFixed(2)}s`;
}

function schedulePlaybackOverlaySync() {
    cancelPlaybackOverlaySync();
    const tick = () => {
        if (player.paused || player.ended) {
            playbackOverlayHandle = null;
            updateTransportButton();
            syncPerceptionUI();
            return;
        }
        syncPerceptionUI();
        if (typeof player.requestVideoFrameCallback === 'function') {
            playbackOverlayHandle = player.requestVideoFrameCallback(() => tick());
        } else {
            playbackOverlayHandle = window.requestAnimationFrame(tick);
        }
    };
    tick();
}

function cancelPlaybackOverlaySync() {
    if (playbackOverlayHandle === null) return;
    if (typeof player.cancelVideoFrameCallback === 'function' && typeof playbackOverlayHandle === 'number') {
        try {
            player.cancelVideoFrameCallback(playbackOverlayHandle);
        } catch (_err) {
            window.cancelAnimationFrame(playbackOverlayHandle);
        }
    } else {
        window.cancelAnimationFrame(playbackOverlayHandle);
    }
    playbackOverlayHandle = null;
}

function stepFrames(frameDelta) {
    player.pause();
    player.currentTime = clampVideoTime(player.currentTime + (frameDelta / 30));
    syncPerceptionUI();
}

function jumpSeconds(secondsDelta) {
    player.pause();
    player.currentTime = clampVideoTime(player.currentTime + secondsDelta);
    syncPerceptionUI();
}

function rewindToStart() {
    player.pause();
    player.currentTime = 0;
    syncPerceptionUI();
}

function togglePlayback() {
    if (player.paused) {
        player.play();
    } else {
        player.pause();
    }
    updateTransportButton();
}

function updateTransportButton() {
    if (!playPauseButton) return;
    playPauseButton.textContent = player.paused ? 'PLAY' : 'PAUSE';
}

function getVisibleJerseyMetadata(detection) {
    const jerseyNumber = detection.identity_jersey_number;
    const jerseyConfidence = detection.identity_jersey_number_confidence;
    const jerseyEvidenceCount = detection.identity_jersey_evidence_count || 0;
    if (jerseyNumber === null || jerseyNumber === undefined || jerseyNumber === '') return null;
    if (jerseyConfidence === null || jerseyConfidence === undefined) return null;
    if (jerseyConfidence < JERSEY_UI_MIN_CONFIDENCE) return null;
    if (jerseyEvidenceCount < JERSEY_UI_MIN_EVIDENCE) return null;
    return {
        number: jerseyNumber,
        confidence: jerseyConfidence,
        evidenceCount: jerseyEvidenceCount,
    };
}

function resetUI() {
    isCalibrating = true;
    updateCalibrationStats();
    document.getElementById('calibration-hint').style.display = 'block';
    
    player.onloadedmetadata = () => {
        syncCanvasSize();
        updateTransportButton();
        syncPerceptionUI();
    };
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    syncPerceptionUI();
}

document.getElementById('btn-reset-cal').addEventListener('click', () => {
    calibrationPoints = [];
    resetUI();
});

document.getElementById('btn-run-cal').addEventListener('click', () => {
    if (calibrationPoints.length < 4) {
        alert("Need at least 4 clicked points for calibration. Corners are optional.");
        return;
    }
    finishCalibration();
});

function populateLandmarkList() {
    landmarkList.innerHTML = '';
    const familyLabels = {
        sideline: 'Sidelines',
        baseline: 'Baselines',
        lane: 'Lane / Paint',
        three_point_arc: '3-Point Arc',
        rim: 'Rims',
    };
    const grouped = {};
    Object.entries(landmarkSpecs).forEach(([landmarkId, spec]) => {
        const family = spec.family || 'other';
        if (!grouped[family]) grouped[family] = [];
        grouped[family].push([landmarkId, spec]);
    });
    Object.entries(grouped).forEach(([family, entries]) => {
        const optgroup = document.createElement('optgroup');
        optgroup.label = familyLabels[family] || family;
        entries.sort((a, b) => a[1].label.localeCompare(b[1].label));
        entries.forEach(([landmarkId, spec]) => {
            const opt = document.createElement('option');
            opt.value = landmarkId;
            opt.textContent = spec.label;
            optgroup.appendChild(opt);
        });
        landmarkList.appendChild(optgroup);
    });
}

function updateCalibrationStats() {
    const uniquePrimitives = new Set(calibrationPoints.map(p => p.primitive_id));
    const uniqueFamilies = new Set(calibrationPoints.map(p => p.primitive_family).filter(Boolean));
    calStats.textContent = `Points: ${calibrationPoints.length} // primitives: ${uniquePrimitives.size} // families: ${uniqueFamilies.size}`;
}

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
            togglePlayback();
            break;
        case 'ArrowRight':
            stepFrames(1);
            break;
        case 'ArrowLeft':
            stepFrames(-1);
            break;
        case 'r':
        case 'R':
            rewindToStart();
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
    
    const primitive_id = landmarkList.value;
    const primitiveSpec = landmarkSpecs[primitive_id] || null;
    const t_ms = Math.floor(player.currentTime * 1000);

    if (clickX >= 0 && clickX <= box.width && clickY >= 0 && clickY <= box.height && primitive_id) {
        calibrationPoints.push({
            point_key: `${primitive_id}_${t_ms}_${calibrationPoints.length}`,
            sample_order: calibrationPoints.length,
            x,
            y,
            primitive_id,
            primitive_family: primitiveSpec ? primitiveSpec.family : null,
            primitive_kind: primitiveSpec ? primitiveSpec.kind : null,
            t_ms
        });
        drawPoint(x, y, primitiveSpec ? primitiveSpec.label : primitive_id);
        updateCalibrationStats();
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
    if (!perceptionData || !perceptionData.enabled) return null;
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

function getCurrentLivePlaySegment(frame) {
    if (!(perceptionData && Array.isArray(perceptionData.live_play_segments) && frame)) return null;
    return perceptionData.live_play_segments.find(
        segment => segment.start_frame <= frame.frame_idx && frame.frame_idx <= segment.end_frame
    ) || null;
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
    const identityLabel = detection.identity_track_id !== null && detection.identity_track_id !== undefined && detection.identity_track_id !== detection.track_id
        ? `/I${detection.identity_track_id}`
        : '';
    const visibleJersey = getVisibleJerseyMetadata(detection);
    const jerseyLabel = visibleJersey
        ? ` #${visibleJersey.number}`
        : '';
    const jerseyConfidenceLabel = visibleJersey
        ? `(${Math.round(visibleJersey.confidence * 100)})`
        : '';
    const uniformLabel = detection.uniform_bucket && detection.uniform_bucket !== 'unknown'
        ? ` ${detection.uniform_bucket.toUpperCase()}`
        : '';
    const confLabel = `${Math.round((detection.confidence || 0) * 100)}%`;
    const activeScore = detection.active_player_score !== undefined && detection.active_player_score !== null
        ? ` A${Math.round(detection.active_player_score * 100)}`
        : '';
    const motionLabel = detection.motion_speed_px !== undefined && detection.motion_speed_px !== null
        ? ` M${detection.motion_speed_px.toFixed(1)}`
        : '';
    const repairLabel = detection.synthesized ? ' SYN' : '';
    const candidateColor = detection.active_player_candidate ? "#35f28b" : "#ff8f70";

    ctx.save();
    ctx.strokeStyle = detection.synthesized ? "#ffd166" : "#3ed8ff";
    ctx.lineWidth = 4;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `${trackLabel}${identityLabel}${jerseyLabel}${jerseyConfidenceLabel}${uniformLabel} ${confLabel}${activeScore}${motionLabel}${repairLabel}`;
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

function drawBallOverlay(ballDetection) {
    if (!ballDetection || !ballDetection.bbox_xyxy) return;
    const [x1, y1, x2, y2] = ballDetection.bbox_xyxy;
    const center = ballDetection.center_xy || [((x1 + x2) / 2), ((y1 + y2) / 2)];
    const confidence = Math.round((ballDetection.confidence || 0) * 100);
    const radius = Math.max(7, Math.min(18, Math.max(x2 - x1, y2 - y1) / 2));
    const label = `BALL ${confidence}%`;
    const labelWidth = Math.max(90, label.length * 10 + 16);
    const labelHeight = 24;
    const labelX = Math.max(10, Math.min(overlay.width - labelWidth - 10, center[0] - labelWidth / 2));
    const labelY = Math.max(10, y1 - labelHeight - 6);

    ctx.save();
    ctx.strokeStyle = "#ffb000";
    ctx.fillStyle = "rgba(255, 176, 0, 0.15)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(center[0], center[1], radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "rgba(51, 36, 0, 0.92)";
    ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
    ctx.strokeStyle = "#ffd166";
    ctx.lineWidth = 2;
    ctx.strokeRect(labelX, labelY, labelWidth, labelHeight);
    ctx.fillStyle = "#fff2c7";
    ctx.font = "bold 15px monospace";
    ctx.fillText(label, labelX + 8, labelY + 16);
    ctx.restore();
}

function redrawOverlay() {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    calibrationPoints.forEach(p => {
        if (Math.abs(p.t_ms - player.currentTime * 1000) < 100) {
            const spec = landmarkSpecs[p.primitive_id];
            drawPoint(p.x, p.y, spec ? spec.label : p.primitive_id);
        }
    });

    const frame = getCurrentPerceptionFrame();
    if (perceptionVisible && frame) {
        frame.detections.forEach(drawDetectionOverlay);
        drawBallOverlay(frame.ball_detection);
        explainerPanel.style.display = 'block';
        explainerTitle.textContent = perceptionData.title || 'Layer 1 Perception Overlay';
        const firstDetection = frame.detections.find(d => d.court_xy);
        const activeCount = frame.detections.filter(d => d.active_player_candidate).length;
        const synthCount = frame.detections.filter(d => d.synthesized).length;
        const repairedIdentityCount = frame.detections.filter(d => d.identity_track_id !== undefined && d.identity_track_id !== null && d.identity_track_id !== d.track_id).length;
        const jerseyCount = frame.detections.filter(d => !!getVisibleJerseyMetadata(d)).length;
        const identityHypothesisCount = Array.isArray(frame.identity_hypothesis_group_ids) ? frame.identity_hypothesis_group_ids.length : 0;
        const ballDetection = frame.ball_detection || null;
        const continuitySegment = frame.continuity_segment_id !== undefined && frame.continuity_segment_id !== null
            ? ` continuity segment ${frame.continuity_segment_id}`
            : '';
        const discontinuityLabel = frame.discontinuity_label || 'continuous';
        const discontinuityScore = frame.discontinuity_score !== undefined && frame.discontinuity_score !== null
            ? frame.discontinuity_score.toFixed(2)
            : 'n/a';
        const ballText = ballDetection
            ? ` Ball ${Math.round((ballDetection.confidence || 0) * 100)}%${ballDetection.court_xy ? ` @ (${ballDetection.court_xy[0].toFixed(1)}, ${ballDetection.court_xy[1].toFixed(1)})` : ''}.`
            : ' Ball not detected in this frame.';
        const livePlayLabel = frame.live_play_label || 'unknown';
        const livePlayScore = frame.live_play_score !== undefined && frame.live_play_score !== null
            ? frame.live_play_score.toFixed(2)
            : 'n/a';
        const segment = getCurrentLivePlaySegment(frame);
        const segmentText = segment
            ? ` Segment ${segment.label} frames ${segment.start_frame}-${segment.end_frame}.`
            : '';
        const dominantSignal = segment && segment.reasons_summary ? segment.reasons_summary.dominant_signal : null;
        const courtText = firstDetection
            ? ` First court point: (${firstDetection.court_xy[0].toFixed(1)}, ${firstDetection.court_xy[1].toFixed(1)}).`
            : '';
        explainerDescription.textContent =
            `Real Ultralytics detections and pose keypoints for frame ${frame.frame_idx}. ` +
            `Green labels mark active-player candidates, orange labels mark likely sidelines/spectators, yellow boxes are repaired track gaps, and orange circles mark the best ball detection. ` +
            `Cyan skeletons show keypoints. Live-play gate: ${livePlayLabel} @ ${livePlayScore}.` +
            ` Continuity: ${discontinuityLabel} @ ${discontinuityScore}.${continuitySegment ? ` In${continuitySegment}.` : ''}` +
            (dominantSignal ? ` Dominant signal: ${dominantSignal}.` : '') +
            segmentText +
            ballText +
            (frame.calibrated ? ` Calibration is active for this frame.${courtText}` : ' No calibration is active for this frame.');
        explainerStatus.textContent =
            `${frame.detections.length} detections // ${activeCount} active candidates // ${synthCount} repaired // ${repairedIdentityCount} identity-bridged // ${identityHypothesisCount} identity-hyp groups // ${jerseyCount} jersey-tagged // continuity ${discontinuityLabel} @ ${discontinuityScore}${continuitySegment ? ` // seg ${frame.continuity_segment_id}` : ''} // ball ${ballDetection ? Math.round((ballDetection.confidence || 0) * 100) + '%' : 'none'} // live ${livePlayLabel} @ ${livePlayScore} // ${Math.round(frame.t_ms / 10) / 100}s // ${perceptionData.model.name} // ${frame.calibrated ? 'calibrated' : 'raw'}`;
    } else if (perceptionVisible && perceptionData && perceptionData.enabled) {
        explainerPanel.style.display = 'block';
        explainerTitle.textContent = perceptionData.title || 'Layer 1 Perception Overlay';
        explainerDescription.textContent = 'Play or step through the clip to inspect the current frame’s tracked detections and keypoints.';
        explainerStatus.textContent = 'No frame-aligned detections at this timestamp';
    } else if (perceptionData && perceptionData.enabled) {
        explainerPanel.style.display = 'block';
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
        const identityLabel = detection.identity_track_id !== null && detection.identity_track_id !== undefined && detection.identity_track_id !== detection.track_id
            ? ` // I${detection.identity_track_id}`
            : '';
        const visibleJersey = getVisibleJerseyMetadata(detection);
        const jerseyLabel = visibleJersey
            ? ` // #${visibleJersey.number}`
            : '';
        const jerseyConfidenceLabel = visibleJersey
            ? ` // jersey ${Math.round(visibleJersey.confidence * 100)}`
            : '';
        const activeLabel = detection.active_player_score !== undefined && detection.active_player_score !== null
            ? ` // active ${Math.round(detection.active_player_score * 100)}`
            : '';
        const motionText = detection.motion_speed_px !== undefined && detection.motion_speed_px !== null
            ? ` // motion ${detection.motion_speed_px.toFixed(1)}`
            : '';
        const appearanceText = detection.appearance_team_distance !== undefined && detection.appearance_team_distance !== null
            ? ` // teamdist ${detection.appearance_team_distance.toFixed(2)}`
            : '';
        const synthLabel = detection.synthesized ? ' // repaired' : '';
        opt.textContent = `Track ${trackId}${identityLabel}${jerseyLabel}${jerseyConfidenceLabel}${uniformLabel} // ${Math.round((detection.confidence || 0) * 100)}%${activeLabel}${motionText}${appearanceText}${synthLabel}`;
        feedbackTrackList.appendChild(opt);
    });
}

function savePerceptionFeedback(issueType, frameLevelOnly = false) {
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
        track_id: frameLevelOnly ? null : (feedbackTrackList.value || null),
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
                    `Artifact ready for ${clip.id}. Toggle the overlay and step through the clip to inspect real detections, pose, identity repairs, jersey evidence, and calibrated court coordinates when available.`;
                const jerseyReady = data.postprocess && data.postprocess.jersey_ocr && data.postprocess.jersey_ocr.reader_available;
                const jerseyCount = data.postprocess && data.postprocess.jersey_ocr ? data.postprocess.jersey_ocr.identity_count_with_consensus : 0;
                const appearanceCount = data.postprocess && data.postprocess.appearance_cue ? data.postprocess.appearance_cue.prototype_count : 0;
                const hypothesisCount = data.postprocess && data.postprocess.identity_hypotheses ? data.postprocess.identity_hypotheses.group_count : 0;
                const continuityCount = Array.isArray(data.continuity_segments) ? data.continuity_segments.length : 0;
                const livePlaySegments = Array.isArray(data.live_play_segments) ? data.live_play_segments : [];
                const liveSegmentCount = livePlaySegments.filter(segment => segment.label === 'live_play').length;
                const deadSegmentCount = livePlaySegments.filter(segment => segment.label === 'dead_ball').length;
                const uncertainSegmentCount = livePlaySegments.filter(segment => segment.label === 'uncertain').length;
                const ballReady = data.postprocess && data.postprocess.live_play_gate ? data.postprocess.live_play_gate.ball_signal_present : false;
                explainerStatus.textContent = `${data.calibration && data.calibration.enabled ? 'Ready // calibration available' : 'Ready // raw perception only'} // jersey OCR ${jerseyReady ? 'experimental' : 'unavailable'} // ${jerseyCount} identity consensuses // show only >=${Math.round(JERSEY_UI_MIN_CONFIDENCE * 100)}% with ${JERSEY_UI_MIN_EVIDENCE}+ votes // ${appearanceCount} appearance prototypes // ${hypothesisCount} identity-hyp groups // ${continuityCount} continuity segments // ball artifact ${ballReady ? 'enabled' : 'unavailable'} // live segments ${liveSegmentCount} // dead segments ${deadSegmentCount} // uncertain ${uncertainSegmentCount}`;
                feedbackStatus.textContent = 'Select a frame and optionally a track, then save structured feedback.';
            } else {
                feedbackStatus.textContent = 'No perception artifact for this clip yet.';
            }
            updateExplainerButton();
            syncPerceptionUI();
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
player.addEventListener('timeupdate', syncPerceptionUI);
player.addEventListener('seeked', syncPerceptionUI);
player.addEventListener('pause', () => {
    cancelPlaybackOverlaySync();
    syncPerceptionUI();
});
player.addEventListener('play', () => {
    updateTransportButton();
    schedulePlaybackOverlaySync();
});

document.getElementById('feedback-fp').addEventListener('click', () => savePerceptionFeedback('false_positive'));
document.getElementById('feedback-fn').addEventListener('click', () => savePerceptionFeedback('false_negative'));
document.getElementById('feedback-merge').addEventListener('click', () => savePerceptionFeedback('merge_error'));
document.getElementById('feedback-track').addEventListener('click', () => savePerceptionFeedback('track_error'));
document.getElementById('feedback-pose').addEventListener('click', () => savePerceptionFeedback('pose_error'));
document.getElementById('feedback-live').addEventListener('click', () => savePerceptionFeedback('live_play', true));
document.getElementById('feedback-dead').addEventListener('click', () => savePerceptionFeedback('dead_ball', true));
document.getElementById('feedback-uncertain').addEventListener('click', () => savePerceptionFeedback('uncertain_play_state', true));
document.getElementById('feedback-note-save').addEventListener('click', () => savePerceptionFeedback('general_note'));
document.getElementById('btn-play-pause').addEventListener('click', togglePlayback);
document.getElementById('btn-rewind').addEventListener('click', rewindToStart);
document.getElementById('btn-step-back').addEventListener('click', () => stepFrames(-1));
document.getElementById('btn-step-forward').addEventListener('click', () => stepFrames(1));
document.getElementById('btn-back-1s').addEventListener('click', () => jumpSeconds(-1.0));
document.getElementById('btn-forward-1s').addEventListener('click', () => jumpSeconds(1.0));

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
    }).then(async (res) => {
        const data = await res.json();
        if (!res.ok) {
            throw new Error(data.reason || data.status || 'calibration_failed');
        }
        return data;
    })
      .then(data => {
          console.log("Panning Calibration Complete. Frames:", data.frames_calibrated);
          const families = (data.primitive_families || []).join(', ') || 'unknown';
          alert(`Calibration saved. Frames: ${data.frames_calibrated}. Families: ${families}.`);
      })
      .catch((error) => {
          isCalibrating = true;
          document.getElementById('calibration-hint').style.display = 'block';
          alert(`Calibration failed: ${error.message}`);
      });
    setTimeout(() => ctx.clearRect(0, 0, overlay.width, overlay.height), 2000);
}
