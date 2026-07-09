import argparse
import os
import cv2
import numpy as np
import pandas as pd
import joblib as jl
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from pitch_dictionary import detect_pitch_label, PITCH_NAMES, PITCH_TYPES
from pitch_tipping_utils import load_alignment_config

mp_pose = mp.solutions.pose

# OVERLAY TOOL
def overlay_videos_from_config(config_path, alpha=0.5, side_by_side=False, export_path=None):
    cfg = load_alignment_config(config_path)

    video_a = cfg["video_a"]
    video_b = cfg["video_b"]
    frame_a = cfg["frame_a"]
    frame_b = cfg["frame_b"]
    dx = cfg.get("offset_x", 0)
    dy = cfg.get("offset_y", 0)

    cap1 = cv2.VideoCapture(video_a)
    cap2 = cv2.VideoCapture(video_b)

    cap1.set(cv2.CAP_PROP_POS_FRAMES, frame_a)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_b)

    framesA = []
    framesB = []

    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 or not ret2:
            break
        framesA.append(f1)
        framesB.append(f2)

    cap1.release()
    cap2.release()

    if not framesA or not framesB:
        print("No frames to overlay.")
        return

    h, w = framesA[0].shape[:2]
    framesB = [cv2.resize(f, (w, h)) for f in framesB]

    writer = None
    if export_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_size = (w * 2, h) if side_by_side else (w, h)
        Path(os.path.dirname(export_path)).mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(export_path, fourcc, 30, out_size)

    idx = 0
    while True:
        f1 = framesA[idx]
        f2 = framesB[idx]

        if side_by_side:
            canvas = cv2.hconcat([f1, f2])
        else:
            canvas = f1.copy()

            y1 = max(0, dy)
            y2 = min(h, dy + h)
            x1 = max(0, dx)
            x2 = min(w, dx + w)

            overlay_region = canvas[y1:y2, x1:x2]
            f2_region = f2[y1 - dy:y2 - dy, x1 - dx:x2 - dx]

            blended = cv2.addWeighted(overlay_region, alpha, f2_region, 1 - alpha, 0)
            canvas[y1:y2, x1:x2] = blended

        cv2.imshow("Pitch Overlay", canvas)

        if writer is not None:
            writer.write(canvas)

        key = cv2.waitKey(30)
        if key == 27:  # ESC
            break
        elif key == ord('+'):
            alpha = min(1.0, alpha + 0.05)
        elif key == ord('-'):
            alpha = max(0.0, alpha - 0.05)

        idx = (idx + 1) % len(framesA)

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

# POSE EXTRACTION
def extract_pose(video_path, pitch_label):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_features = []

    for _ in tqdm(range(total_frames), desc=f"Extracting {os.path.basename(video_path)}"):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            vec = []
            for p in lm:
                vec.extend([p.x, p.y, p.z, p.visibility])
            frame_features.append(vec)

    cap.release()
    pose.close()

    if not frame_features:
        return None

    arr = np.array(frame_features)
    mean_feats = arr.mean(axis=0)
    std_feats = arr.std(axis=0)
    feats = np.concatenate([mean_feats, std_feats])

    return {
        "video_path": video_path,
        "pitch_label": pitch_label,
        "features": feats
    }

# FEATURE SAVING
def save_features(samples, out_path):
    if not samples:
        print("No samples extracted.")
        return

    feat_len = len(samples[0]["features"])
    cols = [f"f_{i}" for i in range(feat_len)]

    rows = []
    for s in samples:
        row = {"video_path": s["video_path"], "pitch_label": s["pitch_label"]}
        for i, v in enumerate(s["features"]):
            row[cols[i]] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved features to {out_path}")

# TIPPING SCORE
def tipping_score(model_path, video_a, video_b, label_a, label_b):
    model = jl.load(model_path)

    sample_a = extract_pose(video_a, label_a)
    sample_b = extract_pose(video_b, label_b)

    Xa = sample_a["features"].reshape(1, -1)
    Xb = sample_b["features"].reshape(1, -1)

    pa = model.predict_proba(Xa)[0]
    pb = model.predict_proba(Xb)[0]

    score = np.abs(pa - pb).sum()
    print(f"Tipping score: {score:.3f}")

# TRAIN MODEL
def train_model(features_csv, model_out):
    df = pd.read_csv(features_csv)
    X = df[[c for c in df.columns if c.startswith("f_")]].values
    y = df["pitch_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    jl.dump(clf, model_out)
    print(f"Saved model to {model_out}")

# CLI
def main():
    parser = argparse.ArgumentParser(description="Pitch tipping tools")
    sub = parser.add_subparsers(dest="cmd")

    # Overlay command
    p_overlay = sub.add_parser("overlay")
    p_overlay.add_argument("--config", required=True, help="Path to alignment JSON file")
    p_overlay.add_argument("--alpha", type=float, default=0.5, help="Overlay opacity")
    p_overlay.add_argument("--side_by_side", action="store_true", help="Side-by-side playback")
    p_overlay.add_argument("--export", type=str, default=None, help="Export overlay to video")

    args = parser.parse_args()

    if args.cmd == "overlay":
        overlay_videos_from_config(
            args.config,
            alpha=args.alpha,
            side_by_side=args.side_by_side,
            export_path=args.export
        )
    else:
        print("No command specified. Use: overlay")

if __name__ == "__main__":
    main()
