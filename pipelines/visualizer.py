import cv2
import json


def draw_game_dna(frame, events, t_ms):
    """
    Overlays Game DNA (boxes, skeletons, labels) onto a video frame.
    """
    # Filter events for the current timestamp (+/- 100ms)
    active = [e for e in events if abs(e.get("t_ms", 0) - t_ms) < 100]

    for ev in active:
        kind = ev.get("kind")
        if kind == "player":
            # Draw BBox
            bx = ev.get("bbox_xywh")
            if bx:
                cx, cy, w, h = bx
                x1, y1 = int(cx - w/2), int(cy - h/2)
                x2, y2 = int(cx + w/2), int(cy + h/2)
                # BGR: Yellow for active players
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                # Label
                label = f"ID:{ev.get('track_id')} {ev.get('action')}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        elif kind == "ball":
            x, y = int(ev.get("x", 0)), int(ev.get("y", 0))
            # Orange for ball
            cv2.circle(frame, (x, y), 8, (0, 165, 255), -1)

    return frame


def visualize_video(video_path, dna_path):
    cap = cv2.VideoCapture(video_path)

    events = []
    with open(dna_path, 'r') as f:
        for line in f:
            events.append(json.loads(line))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        frame = draw_game_dna(frame, events, t_ms)
        cv2.imshow("HoopSense DNA Visualizer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python visualizer.py <video> <dna_jsonl>")
    else:
        visualize_video(sys.argv[1], sys.argv[2])
