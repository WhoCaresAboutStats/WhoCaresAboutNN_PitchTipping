import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from pitch_tipping_utils import (
    trim_video_to_start,
    save_alignment_config
)
import os

st.set_page_config(page_title="Pitch Alignment Selector", layout="wide")
st.title("Pitch Delivery Alignment Selector")

# User Inputs
video_a = st.text_input("Video A path", "Data/Videos/Skenes_FF_000.mp4")
video_b = st.text_input("Video B path", "Data/Videos/Skenes_FF_002.mp4")
config_path = st.text_input("Alignment config output", "data/alignment/skenes_alignment.json")

# Load full videos into memory
def load_all_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append({"index": idx, "frame": frame})
        idx += 1
    cap.release()
    return frames

if st.button("Load Videos"):
    if not Path(video_a).exists() or not Path(video_b).exists():
        st.error("One or both video paths do not exist.")
    else:
        st.session_state.frames_a = load_all_frames(video_a)
        st.session_state.frames_b = load_all_frames(video_b)
        st.session_state.selected_a = None
        st.session_state.selected_b = None
        st.session_state.offset_x = 0
        st.session_state.offset_y = 0
        st.session_state.zoom = 1.0
        st.session_state.opacity = 0.5

# Drag through Video A
if "frames_a" in st.session_state and st.session_state.frames_a:
    st.subheader("Drag Through ALL Frames of Video A")

    max_idx_a = len(st.session_state.frames_a) - 1
    idx_a = st.slider("Frame index for Video A", 0, max_idx_a, 0)

    frameA = st.session_state.frames_a[idx_a]["frame"]
    frameA_rgb = cv2.cvtColor(frameA, cv2.COLOR_BGR2RGB)

    st.image(frameA_rgb, caption=f"Video A – Frame {idx_a}")

    if st.button("Confirm Frame A"):
        st.session_state.selected_a = idx_a
        st.success(f"Selected frame A: {idx_a}")

# Drag through Video B
if "frames_b" in st.session_state and st.session_state.frames_b:
    st.subheader("Drag Through ALL Frames of Video B")

    max_idx_b = len(st.session_state.frames_b) - 1
    idx_b = st.slider("Frame index for Video B", 0, max_idx_b, 0)

    frameB = st.session_state.frames_b[idx_b]["frame"]
    frameB_rgb = cv2.cvtColor(frameB, cv2.COLOR_BGR2RGB)

    st.image(frameB_rgb, caption=f"Video B – Frame {idx_b}")

    if st.button("Confirm Frame B"):
        st.session_state.selected_b = idx_b
        st.success(f"Selected frame B: {idx_b}")
import cv2

def play_pause_preview(framesA, framesB):
    idx = 0
    paused = False
    total = min(len(framesA), len(framesB))

    while True:
        frameA = framesA[idx]["frame"]
        frameB = framesB[idx]["frame"]

        # Convert for display
        fA = cv2.cvtColor(frameA, cv2.COLOR_BGR2RGB)
        fB = cv2.cvtColor(frameB, cv2.COLOR_BGR2RGB)

        # Side-by-side preview
        combined = cv2.hconcat([frameA, frameB])
        cv2.imshow("Alignment Preview (SPACE=Pause, ←/→ Step, ESC Quit)", combined)

        key = cv2.waitKey(30)

        if key == 27:  # ESC
            break

        elif key == ord(' '):  # SPACE
            paused = not paused

        elif key == 81:  # LEFT ARROW
            paused = True
            idx = max(0, idx - 1)

        elif key == 83:  # RIGHT ARROW
            paused = True
            idx = min(total - 1, idx + 1)

        if not paused:
            idx = (idx + 1) % total

    cv2.destroyAllWindows()
    return idx

# Side-by-side confirmation
if st.session_state.get("selected_a") is not None and st.session_state.get("selected_b") is not None:
    st.subheader("Confirm the two selected frames")

    frameA = st.session_state.frames_a[st.session_state.selected_a]["frame"]
    frameB = st.session_state.frames_b[st.session_state.selected_b]["frame"]

    frameA_rgb = cv2.cvtColor(frameA, cv2.COLOR_BGR2RGB)
    frameB_rgb = cv2.cvtColor(frameB, cv2.COLOR_BGR2RGB)

    c1, c2 = st.columns(2)
    c1.image(frameA_rgb, caption=f"Video A – Frame {st.session_state.selected_a}")
    c2.image(frameB_rgb, caption=f"Video B – Frame {st.session_state.selected_b}")

# Alignment controls + overlay preview
if st.session_state.get("selected_a") is not None and st.session_state.get("selected_b") is not None:
    st.subheader("Alignment Controls + Overlay Preview")

    frameA = st.session_state.frames_a[st.session_state.selected_a]["frame"]
    frameB = st.session_state.frames_b[st.session_state.selected_b]["frame"]

    frameA = cv2.cvtColor(frameA, cv2.COLOR_BGR2RGB)
    frameB = cv2.cvtColor(frameB, cv2.COLOR_BGR2RGB)

    h, w, _ = frameA.shape

    # Zoom
    st.session_state.zoom = st.number_input(
        "Zoom (scale for B)", min_value=0.5, max_value=3.0,
        value=st.session_state.zoom, step=0.05
    )

    # Opacity
    st.session_state.opacity = st.number_input(
        "Overlay opacity", min_value=0.0, max_value=1.0,
        value=st.session_state.opacity, step=0.05
    )

    # Offsets
    st.session_state.offset_x = st.number_input(
        "Offset X (pixels)", min_value=-w, max_value=w,
        value=st.session_state.offset_x, step=1
    )
    st.session_state.offset_y = st.number_input(
        "Offset Y (pixels)", min_value=-h, max_value=h,
        value=st.session_state.offset_y, step=1
    )

    # Apply zoom
    zoom_w = int(w * st.session_state.zoom)
    zoom_h = int(h * st.session_state.zoom)
    frameB_zoom = cv2.resize(frameB, (zoom_w, zoom_h))

    # Overlay preview
    canvas = frameA.copy()

    dx = st.session_state.offset_x
    dy = st.session_state.offset_y

    x1 = max(0, dx)
    y1 = max(0, dy)
    x2 = min(w, dx + zoom_w)
    y2 = min(h, dy + zoom_h)

    bx1 = x1 - dx
    by1 = y1 - dy
    bx2 = bx1 + (x2 - x1)
    by2 = by1 + (y2 - y1)

    if x2 > x1 and y2 > y1:
        overlay_region = canvas[y1:y2, x1:x2]
        b_region = frameB_zoom[by1:by2, bx1:bx2]
        blended = cv2.addWeighted(
            overlay_region,
            1.0 - st.session_state.opacity,
            b_region,
            st.session_state.opacity,
            0
        )
        canvas[y1:y2, x1:x2] = blended

    st.image(canvas, caption="Overlay preview")


    # Save Alignment

    if st.button("Save Alignment"):
        save_alignment_config(
            config_path,
            video_a,
            video_b,
            st.session_state.selected_a,
            st.session_state.selected_b,
            st.session_state.offset_x,
            st.session_state.offset_y
        )
        st.success(f"Saved alignment to {config_path}")


    # Trim Videos

    trim_dir = st.text_input("Trimmed videos output dir", "Data/Aligned")

    if st.button("Trim Videos"):
        out_a = str(Path(trim_dir) / Path(video_a).name.replace(".mp4", "_aligned.mp4"))
        out_b = str(Path(trim_dir) / Path(video_b).name.replace(".mp4", "_aligned.mp4"))

        trim_video_to_start(video_a, st.session_state.selected_a, out_a)
        trim_video_to_start(video_b, st.session_state.selected_b, out_b)

        st.success(f"Trimmed videos saved:\n{out_a}\n{out_b}")

# Quit Button
st.write("---")
if st.button("Quit Alignment Tool"):
    st.success("Closing the alignment tool…")
    os._exit(0)
