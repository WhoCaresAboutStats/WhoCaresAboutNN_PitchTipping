import cv2
import numpy as np
import json
import os

def extract_preview_frames(video_path, num_frames=9, middle_fraction=0.33):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from {video_path}")

    # Middle third of the video
    start = int(total * (0.5 - middle_fraction / 2))
    end = int(total * (0.5 + middle_fraction / 2))

    indices = np.linspace(start, end, num_frames).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append({"index": int(idx), "frame": frame})

    cap.release()
    return frames

def trim_video_to_start(video_path, start_frame, out_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

def save_alignment_config(path, video_a, video_b, frame_a, frame_b, offset_x=0, offset_y=0):
    cfg = {
        "video_a": video_a,
        "video_b": video_b,
        "frame_a": int(frame_a),
        "frame_b": int(frame_b),
        "offset_x": int(offset_x),
        "offset_y": int(offset_y)
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_alignment_config(path):
    with open(path, "r") as f:
        return json.load(f)
