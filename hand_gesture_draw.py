"""
hand_gesture_draw.py
─────────────────────────────────────────────────────────────────
Runs on the DRAWER's laptop.
- Uses webcam + MediaPipe to detect hand landmarks
- Tracks index finger tip (landmark 8) for drawing
- Uses KNN model to classify gestures:
    ✌️  "draw"  mode  → finger traces on canvas
    ✊  "stop"  mode  → finger lifted, no drawing
    🖐  "clear" mode  → clears the canvas
- Sends drawing coordinates via Socket.IO to the server
─────────────────────────────────────────────────────────────────
USAGE:
    python hand_gesture_draw.py --server http://localhost:5000 --name Alice
"""

import cv2
import mediapipe as mp
import numpy as np
import socketio
import argparse
import time
import os

# ── Try importing sklearn; fall back to rule-based if not trained ──
try:
    import joblib
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "ml", "gesture_model.pkl")
    SCALER_PATH = os.path.join(os.path.dirname(__file__), "ml", "scaler.pkl")
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        gesture_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        USE_MODEL = True
        print("[INFO] KNN gesture model loaded.")
    else:
        USE_MODEL = False
        print("[INFO] No trained model found. Using rule-based gesture detection.")
except ImportError:
    USE_MODEL = False
    print("[INFO] scikit-learn not installed. Using rule-based gesture detection.")

# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--server', default='http://localhost:5000', help='Server URL')
parser.add_argument('--name', default='Drawer', help='Your player name')
parser.add_argument('--cam', default=0, type=int, help='Camera index')
args = parser.parse_args()

# ─────────────────────────────────────────────
# SOCKET.IO CLIENT
# ─────────────────────────────────────────────
sio = socketio.Client()
connected = False
current_word = ""
my_score = 0
round_info = ""

@sio.event
def connect():
    global connected
    connected = True
    print(f"[SOCKET] Connected to {args.server}")
    sio.emit('join_game', {'name': args.name})

@sio.event
def disconnect():
    global connected
    connected = False
    print("[SOCKET] Disconnected from server.")

@sio.on('joined')
def on_joined(data):
    print(f"[GAME] Joined as {data['role'].upper()} | Players: {data['player_count']}/2")

@sio.on('round_start')
def on_round_start(data):
    global current_word, round_info
    current_word = data.get('word', '')
    round_info = f"Round {data['round']} | Word: {current_word.upper()}"
    print(f"\n[ROUND] {round_info}")

@sio.on('round_end')
def on_round_end(data):
    global round_info, my_score, current_word
    winner = data.get('winner', 'No one')
    word = data.get('word', '?')
    scores = data.get('scores', {})
    score_str = " | ".join([f"{k}: {v}" for k, v in scores.items()])
    my_score = scores.get(args.name, 0)
    round_info = f"Word was: {word.upper()} | Winner: {winner} | {score_str}"
    print(f"[ROUND END] {round_info}")
    current_word = ""

@sio.on('guess_broadcast')
def on_guess(data):
    print(f"[GUESS] {data['name']}: {data['guess']}")

@sio.on('error')
def on_error(data):
    print(f"[ERROR] {data['message']}")

# ─────────────────────────────────────────────
# MEDIAPIPE SETUP
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# ─────────────────────────────────────────────
# FEATURE EXTRACTION (for KNN model)
# ─────────────────────────────────────────────
def extract_features(landmarks):
    """
    Extracts 42 normalized features from 21 hand landmarks.
    Each landmark → (x, y) relative to wrist (landmark 0).
    Normalized by hand span for scale invariance.
    """
    wrist = landmarks[0]
    coords = [(lm.x - wrist.x, lm.y - wrist.y) for lm in landmarks]
    flat = [v for pair in coords for v in pair]

    # Normalize by max distance (scale invariance)
    max_dist = max(abs(v) for v in flat) or 1.0
    normalized = [v / max_dist for v in flat]
    return np.array(normalized, dtype=np.float32)

# ─────────────────────────────────────────────
# RULE-BASED GESTURE DETECTION (fallback)
# ─────────────────────────────────────────────
def fingers_up(landmarks):
    """Returns list of which fingers are extended."""
    tips = [8, 12, 16, 20]   # Index, Middle, Ring, Pinky tips
    pips = [6, 10, 14, 18]   # Corresponding PIP joints
    up = []
    for tip, pip in zip(tips, pips):
        up.append(landmarks[tip].y < landmarks[pip].y)
    # Thumb (compare x instead of y for typical hand orientation)
    thumb_up = landmarks[4].x < landmarks[3].x
    return [thumb_up] + up  # [thumb, index, middle, ring, pinky]

def rule_based_gesture(landmarks):
    """
    draw  → only index finger up
    stop  → no fingers up (fist)
    clear → all 5 fingers up (open palm)
    """
    up = fingers_up(landmarks)
    if all(up):
        return "clear"
    elif up[1] and not up[2] and not up[3] and not up[4]:
        return "draw"
    else:
        return "stop"

def classify_gesture(landmarks):
    if USE_MODEL:
        features = extract_features(landmarks).reshape(1, -1)
        features_scaled = scaler.transform(features)
        return gesture_model.predict(features_scaled)[0]
    else:
        return rule_based_gesture(landmarks)

# ─────────────────────────────────────────────
# DRAWING CANVAS
# ─────────────────────────────────────────────
CANVAS_W, CANVAS_H = 640, 480
canvas = np.ones((CANVAS_H, CANVAS_W, 3), dtype=np.uint8) * 255
prev_x, prev_y = None, None
last_send_time = 0
SEND_INTERVAL = 0.03   # ~33fps transmission

GESTURE_COLORS = {
    "draw":  (30, 30, 220),    # Blue
    "stop":  (200, 200, 200),  # Grey
    "clear": (50, 200, 50),    # Green
}

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    global canvas, prev_x, prev_y, last_send_time

    # Connect to server
    try:
        sio.connect(args.server)
    except Exception as e:
        print(f"[ERROR] Cannot connect to server: {e}")
        print(f"  Make sure the server is running at {args.server}")
        return

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sio.disconnect()
        return

    print("\n[CONTROLS]")
    print("  ☝️  Index finger only  → DRAW")
    print("  ✊  Fist               → STOP drawing")
    print("  🖐  Open palm          → CLEAR canvas")
    print("  Q key                  → Quit\n")

    gesture = "stop"
    gesture_display_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display = frame.copy()
        h, w = frame.shape[:2]

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(display, hand_lms, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_lms.landmark
            gesture = classify_gesture(landmarks)
            gesture_display_timer = 20  # frames to show gesture label

            # Index finger tip position
            tip = landmarks[8]
            fx = int(tip.x * CANVAS_W)
            fy = int(tip.y * CANVAS_H)

            if gesture == "draw":
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (fx, fy), (30, 30, 220), 5)

                    # Transmit to server
                    now = time.time()
                    if now - last_send_time > SEND_INTERVAL and connected:
                        sio.emit('draw_data', {
                            'x1': prev_x / CANVAS_W,
                            'y1': prev_y / CANVAS_H,
                            'x2': fx / CANVAS_W,
                            'y2': fy / CANVAS_H,
                            'color': '#1e1edc',
                            'width': 5
                        })
                        last_send_time = now

                prev_x, prev_y = fx, fy

                # Draw circle at fingertip
                cv2.circle(display, (int(tip.x * w), int(tip.y * h)), 10,
                           GESTURE_COLORS["draw"], -1)

            elif gesture == "clear":
                canvas[:] = 255
                prev_x, prev_y = None, None
                if connected:
                    sio.emit('clear_canvas', {})

            else:  # stop
                prev_x, prev_y = None, None
                cv2.circle(display, (int(tip.x * w), int(tip.y * h)), 10,
                           GESTURE_COLORS["stop"], -1)
        else:
            prev_x, prev_y = None, None

        # ── Overlay: gesture label ──
        if gesture_display_timer > 0:
            label = gesture.upper()
            color = GESTURE_COLORS.get(gesture, (255, 255, 255))
            cv2.putText(display, f"Gesture: {label}", (10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
            gesture_display_timer -= 1

        # ── Overlay: word to draw ──
        if current_word:
            cv2.putText(display, f"DRAW: {current_word.upper()}", (10, h - 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 140, 0), 2)

        # ── Overlay: score ──
        cv2.putText(display, f"Score: {my_score}", (w - 160, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)

        # ── Overlay: connection status ──
        status_color = (0, 200, 0) if connected else (0, 0, 220)
        status_text = "LIVE" if connected else "DISCONNECTED"
        cv2.putText(display, status_text, (w - 100, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Show webcam feed and canvas side by side
        cv2.imshow("Drawer - Webcam", display)
        cv2.imshow("Drawer - Canvas", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if connected:
        sio.disconnect()

if __name__ == '__main__':
    main()
