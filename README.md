# 🎨 Scribble AI — Hand Gesture Multiplayer Drawing Game

A real-time multiplayer drawing-and-guessing game where the Drawer uses **hand gestures via webcam** to draw, and the Guesser types guesses in a browser.

---

## 📁 Project Structure

```
scribble-game/
├── server.py               ← Flask + Socket.IO game server
├── hand_gesture_draw.py    ← Drawer's script (runs on their laptop)
├── collect_dataset.py      ← Collect gesture training data
├── train_model.py          ← Train KNN classifier
├── requirements.txt
├── templates/
│   ├── index.html          ← Landing/lobby page
│   ├── drawer.html         ← Drawer's browser view
│   └── guesser.html        ← Guesser's browser view
├── ml/                     ← Trained model saved here
│   ├── gesture_model.pkl
│   └── scaler.pkl
└── data/
    └── gesture_dataset.csv ← Collected gesture data
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional but recommended) Train the gesture model

**Step 1 — Collect gesture data:**
```bash
python collect_dataset.py
```
- Press `D` → record **DRAW** gesture (index finger pointing up)
- Press `S` → record **STOP** gesture (closed fist)
- Press `C` → record **CLEAR** gesture (open palm)
- Collect ~200 samples per gesture, then press `Q`

**Step 2 — Train the KNN model:**
```bash
python train_model.py
```
This outputs `ml/gesture_model.pkl` and `ml/scaler.pkl`.

> **No model?** The system automatically falls back to rule-based gesture detection using finger landmark positions. Works fine for demos.

---

## 🚀 Running the Game

### Step 1 — Start the server (on one machine)
```bash
python server.py
```
Server runs at `http://0.0.0.0:5000`

### Step 2 — Open browser pages

**Drawer's laptop:**
```
http://<server-ip>:5000/drawer
```

**Guesser's laptop:**
```
http://<server-ip>:5000/guesser
```

### Step 3 — Launch the drawing script (Drawer's laptop)
```bash
python hand_gesture_draw.py --server http://<server-ip>:5000 --name Alice
```

---

## 🎮 How to Play

| Role   | What They Do                                           |
|--------|--------------------------------------------------------|
| Drawer | Sees the secret word, draws it using hand gestures     |
| Guesser| Watches the drawing appear, types guesses in browser   |

**Gestures:**

| Gesture      | Hand Shape          | Action              |
|--------------|---------------------|---------------------|
| ☝️ Draw      | Index finger up     | Trace drawing path  |
| ✊ Stop       | Closed fist         | Pause drawing       |
| 🖐 Clear     | Open palm (5 fingers)| Wipe the canvas    |

**Game Rules:**
- Correct guess → Guesser gets 1 point, round ends
- Time runs out → No points, round ends
- After each round, roles swap automatically
- First to accumulate most points wins!

---

## 🤖 ML Pipeline

```
Webcam Frame
    ↓
MediaPipe Hands  (detects 21 landmarks per hand)
    ↓
Feature Extraction  (42 features: normalized x,y per landmark)
    ↓
KNN Classifier  (k=best, euclidean distance, distance-weighted)
    ↓
Gesture Label: "draw" / "stop" / "clear"
    ↓
Drawing Logic + WebSocket Transmission
```

**Feature Engineering:**
- 21 landmarks × (x, y) = 42 raw values
- Translated relative to wrist (landmark 0) for position invariance
- Normalized by max distance for scale invariance

**KNN Hyperparameters:**
- K: auto-selected via 5-fold cross-validation (1–15)
- Weights: distance-weighted (closer neighbors have more influence)
- Metric: Euclidean distance

---

## 🛠️ System Pipeline

```
Camera → Hand Detection → Landmark Extraction
    → Feature Extraction → KNN Classification
    → Gesture Decision → Drawing Coordinates
    → WebSocket (Socket.IO) → Remote Canvas Rendering
    → Guess Input → Game Logic → Score Update
```

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Cannot connect to server` | Make sure server.py is running and firewall allows port 5000 |
| Webcam not opening | Try `--cam 1` instead of default 0 |
| High latency | Both machines should be on same LAN |
| Gesture not detected | Ensure good lighting; keep hand within frame |
| `mediapipe` install error | Try `pip install mediapipe==0.10.0` |

---

## 📚 Viva Key Points

1. **Why KNN?** — Simple, interpretable, no training time, works well for small datasets with meaningful feature spaces
2. **Why 42 features?** — 21 landmarks × 2 coordinates; normalized for invariance
3. **Why WebSockets?** — Full-duplex, low-latency; better than HTTP polling for real-time drawing
4. **Why MediaPipe?** — Optimized, runs on CPU, pre-trained landmark detection
5. **Why index finger (landmark 8)?** — Topmost point of the pointing gesture; most natural drawing pointer
