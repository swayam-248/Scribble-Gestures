"""
collect_dataset.py
─────────────────────────────────────────────────────────────────
Collects hand landmark data from webcam for gesture classification.
Creates a CSV dataset used to train the KNN model.

GESTURES TO COLLECT:
    draw  → only index finger up (pointing)
    stop  → fist (all fingers closed)
    clear → open palm (all 5 fingers up)

USAGE:
    python collect_dataset.py

CONTROLS:
    D → Record "draw" gesture samples
    S → Record "stop" gesture samples
    C → Record "clear" gesture samples
    Q → Quit and save
─────────────────────────────────────────────────────────────────
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "gesture_dataset.csv")
SAMPLES_PER_SESSION = 200   # samples to collect per key press session
GESTURES = ["draw", "stop", "clear"]

# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

def extract_features(landmarks):
    """
    Returns 42 normalized landmark features (x,y for each of 21 points).
    Relative to wrist (landmark 0), normalized by max distance.
    """
    wrist = landmarks[0]
    coords = [(lm.x - wrist.x, lm.y - wrist.y) for lm in landmarks]
    flat = [v for pair in coords for v in pair]
    max_dist = max(abs(v) for v in flat) or 1.0
    return [v / max_dist for v in flat]

def count_existing_samples():
    """Count samples per class already in dataset."""
    counts = {g: 0 for g in GESTURES}
    if not os.path.exists(DATASET_PATH):
        return counts
    with open(DATASET_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[-1] in counts:
                counts[row[-1]] += 1
    return counts

def main():
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

    # Create CSV with header if it doesn't exist
    write_header = not os.path.exists(DATASET_PATH)
    csvfile = open(DATASET_PATH, 'a', newline='')
    writer = csv.writer(csvfile)
    if write_header:
        header = [f'f{i}' for i in range(42)] + ['label']
        writer.writerow(header)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    recording = False
    current_label = None
    sample_count = 0
    session_samples = 0

    print("\n" + "="*50)
    print("  GESTURE DATASET COLLECTOR")
    print("="*50)
    print("  D → Record DRAW gesture (index finger up)")
    print("  S → Record STOP gesture (fist)")
    print("  C → Record CLEAR gesture (open palm)")
    print("  Q → Quit and save\n")

    counts = count_existing_samples()
    print("  Existing samples:", counts)
    print("="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        display = frame.copy()
        h, w = display.shape[:2]

        hand_detected = False

        if results.multi_hand_landmarks:
            hand_lms = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(display, hand_lms, mp_hands.HAND_CONNECTIONS)
            hand_detected = True

            if recording and current_label:
                features = extract_features(hand_lms.landmark)
                writer.writerow(features + [current_label])
                csvfile.flush()
                sample_count += 1
                session_samples += 1

                if session_samples >= SAMPLES_PER_SESSION:
                    recording = False
                    session_samples = 0
                    counts[current_label] = counts.get(current_label, 0) + SAMPLES_PER_SESSION
                    print(f"  ✓ Collected {SAMPLES_PER_SESSION} samples for '{current_label}'")
                    print(f"  Total: {counts}")
                    current_label = None

        # ── Status overlay ──
        status_color = (0, 200, 0) if recording else (200, 200, 200)
        if recording:
            label_text = f"RECORDING: {current_label.upper()} [{session_samples}/{SAMPLES_PER_SESSION}]"
            cv2.putText(display, label_text, (10, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 220), 2)
            # Progress bar
            prog = int((session_samples / SAMPLES_PER_SESSION) * (w - 40))
            cv2.rectangle(display, (20, 60), (20 + prog, 80), (0, 200, 0), -1)
            cv2.rectangle(display, (20, 60), (w - 20, 80), (100, 100, 100), 2)
        else:
            cv2.putText(display, "PRESS D/S/C TO RECORD | Q TO QUIT", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Hand detection indicator
        det_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        det_text = "HAND DETECTED" if hand_detected else "NO HAND"
        cv2.putText(display, det_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, det_color, 2)

        # Sample counts
        counts_text = f"draw:{counts['draw']} stop:{counts['stop']} clear:{counts['clear']}"
        cv2.putText(display, counts_text, (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 100), 1)

        cv2.imshow("Dataset Collector", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d') or key == ord('D'):
            recording = True
            current_label = "draw"
            session_samples = 0
            print("\n  → Recording DRAW (index finger pointing)...")
        elif key == ord('s') or key == ord('S'):
            recording = True
            current_label = "stop"
            session_samples = 0
            print("\n  → Recording STOP (closed fist)...")
        elif key == ord('c') or key == ord('C'):
            recording = True
            current_label = "clear"
            session_samples = 0
            print("\n  → Recording CLEAR (open palm)...")
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csvfile.close()
    print(f"\n  Dataset saved to: {DATASET_PATH}")
    print(f"  Total samples: {count_existing_samples()}")

if __name__ == '__main__':
    main()
