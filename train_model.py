"""
train_model.py
─────────────────────────────────────────────────────────────────
Trains a K-Nearest Neighbors classifier on the collected gesture
dataset and saves the model + scaler for use in hand_gesture_draw.py

USAGE:
    python train_model.py

OUTPUT:
    ml/gesture_model.pkl   ← Trained KNN classifier
    ml/scaler.pkl          ← StandardScaler (for normalization)
─────────────────────────────────────────────────────────────────
"""

import os
import csv
import numpy as np

def main():
    # ── Late imports (so script fails gracefully with clear message) ──
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report, confusion_matrix
        import joblib
    except ImportError:
        print("[ERROR] scikit-learn is not installed.")
        print("  Run: pip install scikit-learn joblib")
        return

    DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "gesture_dataset.csv")
    MODEL_DIR    = os.path.join(os.path.dirname(__file__), "ml")
    MODEL_PATH   = os.path.join(MODEL_DIR, "gesture_model.pkl")
    SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")

    print("\n" + "="*55)
    print("  KNN GESTURE MODEL TRAINER")
    print("="*55)

    # ── Load dataset ──
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset not found at: {DATASET_PATH}")
        print("  Run collect_dataset.py first.")
        return

    X, y = [], []
    label_counts = {}

    with open(DATASET_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) != 43:
                continue
            features = [float(v) for v in row[:42]]
            label = row[42]
            X.append(features)
            y.append(label)
            label_counts[label] = label_counts.get(label, 0) + 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"\n  Dataset loaded: {len(X)} samples")
    for label, count in sorted(label_counts.items()):
        print(f"    {label:10s} → {count} samples")

    if len(X) < 30:
        print("\n[WARNING] Very few samples. Collect at least 200 per gesture.")
        return

    # ── Preprocessing ──
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Train set: {len(X_train)} | Test set: {len(X_test)}")

    # ── Find best K ──
    print("\n  Finding optimal K...")
    best_k, best_score = 3, 0
    for k in range(1, 16):
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        print(f"    K={k:2d} → CV Accuracy: {mean_score*100:.1f}% (±{scores.std()*100:.1f}%)")
        if mean_score > best_score:
            best_score = mean_score
            best_k = k

    print(f"\n  ✓ Best K = {best_k} (CV Accuracy: {best_score*100:.1f}%)")

    # ── Train final model ──
    model = KNeighborsClassifier(
        n_neighbors=best_k,
        weights='distance',
        metric='euclidean'
    )
    model.fit(X_train, y_train)

    # ── Evaluate ──
    test_accuracy = model.score(X_test, y_test)
    print(f"\n  Test Accuracy: {test_accuracy*100:.1f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, model.predict(X_test)))

    print("\n  Confusion Matrix:")
    classes = sorted(set(y))
    cm = confusion_matrix(y_test, model.predict(X_test), labels=classes)
    # Pretty print confusion matrix
    header = "         " + "  ".join(f"{c:8s}" for c in classes)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"  {classes[i]:7s}" + "  ".join(f"{v:8d}" for v in row)
        print(row_str)

    # ── Save model ──
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n  ✓ Model saved → {MODEL_PATH}")
    print(f"  ✓ Scaler saved → {SCALER_PATH}")
    print("\n  You can now run: python hand_gesture_draw.py")
    print("="*55 + "\n")

if __name__ == '__main__':
    main()
