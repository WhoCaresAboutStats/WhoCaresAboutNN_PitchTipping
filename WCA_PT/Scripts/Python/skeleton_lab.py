import argparse
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

from pitch_tipping_utils import load_alignment_config  # optional, for alignment JSON

mp_pose = mp.solutions.pose
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Utility
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

# Skeleton drawing (Matplotlib)
def draw_skeleton_matplotlib(image, landmarks, save_path=None, show=False):
    plt.figure(figsize=(6, 10))
    plt.imshow(image)
    plt.axis('off')

    # joints
    xs, ys = [], []
    for lm in landmarks:
        x = lm.x * image.shape[1]
        y = lm.y * image.shape[0]
        xs.append(x)
        ys.append(y)
        plt.scatter(x, y, c='red', s=20)

    # bones
    for c in POSE_CONNECTIONS:
        start = landmarks[c[0]]
        end = landmarks[c[1]]
        x1 = start.x * image.shape[1]
        y1 = start.y * image.shape[0]
        x2 = end.x * image.shape[1]
        y2 = end.y * image.shape[0]
        plt.plot([x1, x2], [y1, y2], c='yellow', linewidth=2)

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

# Skeleton drawing (OpenCV overlay)
def draw_skeleton_opencv(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]

    # joints
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    # bones
    for c in POSE_CONNECTIONS:
        s = landmarks[c[0]]
        e = landmarks[c[1]]
        x1 = int(s.x * w)
        y1 = int(s.y * h)
        x2 = int(e.x * w)
        y2 = int(e.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    return frame

# 3D skeleton plotting
def plot_3d_skeleton(landmarks, save_path=None, show=False):
    xs, ys, zs = [], [], []
    for lm in landmarks:
        xs.append(lm.x)
        ys.append(lm.y)
        zs.append(lm.z)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, c='red', s=20)

    for c in POSE_CONNECTIONS:
        s = landmarks[c[0]]
        e = landmarks[c[1]]
        ax.plot(
            [s.x, e.x],
            [s.y, e.y],
            [s.z, e.z],
            c='yellow'
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

# Kinematics (velocity / acceleration)
def compute_kinematics(landmark_series, fps):
    """
    landmark_series: list of [N_landmarks x 3] arrays (x,y,z)
    fps: frames per second
    Returns: dict of per-landmark velocity and acceleration magnitudes
    """
    coords = np.array(landmark_series)
    dt = 1.0 / fps

    vel = np.diff(coords, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt

    vel_mag = np.linalg.norm(vel, axis=2)
    acc_mag = np.linalg.norm(acc, axis=2)

    return {
        "velocity": vel_mag,
        "acceleration": acc_mag
    }

# Main processing
def process_video(
    video_path,
    out_dir,
    save_png=False,
    save_video=False,
    overlay=False,
    side_by_side=False,
    plot_3d_flag=False,
    compute_kinematics_flag=False,
    video_b_path=None,
    config_path=None
):
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Optional second video for alignment
    cap_b = None
    cfg = None
    if video_b_path and config_path:
        cfg = load_alignment_config(config_path)
        cap_b = cv2.VideoCapture(video_b_path)
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, cfg["frame_b"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, cfg["frame_a"])

    # Video writer
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ret, frame0 = cap.read()
        if not ret:
            print("Could not read first frame.")
            return
        h, w = frame0.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if side_by_side:
            out_size = (w * 2, h)
        else:
            out_size = (w, h)
        writer = cv2.VideoWriter(
            str(Path(out_dir) / "skeleton_video.mp4"),
            fourcc,
            fps,
            out_size
        )

    landmark_series = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark

            # store for kinematics
            if compute_kinematics_flag:
                coords = np.array([[lm.x, lm.y, lm.z] for lm in lms])
                landmark_series.append(coords)

            # PNG skeleton
            if save_png:
                png_path = Path(out_dir) / f"skeleton_{frame_idx:05d}.png"
                draw_skeleton_matplotlib(rgb, lms, save_path=str(png_path), show=False)

            # 3D skeleton
            if plot_3d_flag and frame_idx % 10 == 0:
                plot_path = Path(out_dir) / f"skeleton_3d_{frame_idx:05d}.png"
                plot_3d_skeleton(lms, save_path=str(plot_path), show=False)

            # Overlay / side-by-side
            if overlay or side_by_side or writer is not None:
                overlay_frame = frame.copy()
                overlay_frame = draw_skeleton_opencv(overlay_frame, lms, color=(0, 255, 0))

                if cap_b is not None and cfg is not None:
                    # aligned second video frame
                    ret_b, frame_b = cap_b.read()
                    if not ret_b:
                        frame_b = np.zeros_like(frame)
                    frame_b_overlay = frame_b.copy()
                    frame_b_overlay = draw_skeleton_opencv(frame_b_overlay, lms, color=(255, 0, 0))

                    if side_by_side:
                        canvas = cv2.hconcat([overlay_frame, frame_b_overlay])
                    else:
                        canvas = overlay_frame
                else:
                    if side_by_side:
                        canvas = cv2.hconcat([frame, overlay_frame])
                    else:
                        canvas = overlay_frame

                if writer is not None:
                    writer.write(canvas)

                if overlay or side_by_side:
                    cv2.imshow("Skeleton Overlay", canvas)
                    key = cv2.waitKey(1)
                    if key == 27:
                        break

        frame_idx += 1

    cap.release()
    if cap_b is not None:
        cap_b.release()
    pose.close()

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if compute_kinematics_flag and landmark_series:
        kin = compute_kinematics(landmark_series, fps)
        vel = kin["velocity"]  # [T-1, N]
        acc = kin["acceleration"]  # [T-2, N]

        np.save(Path(out_dir) / "velocity.npy", vel)
        np.save(Path(out_dir) / "acceleration.npy", acc)

        # CSV EXPORT
        # Landmarks CSV
        landmark_arr = np.array(landmark_series)  # [T, N, 3]
        landmark_flat = landmark_arr.reshape(landmark_arr.shape[0], -1)
        np.savetxt(Path(out_dir) / "landmarks.csv", landmark_flat, delimiter=",")

        # Velocity CSV
        vel_flat = vel.reshape(vel.shape[0], -1)
        np.savetxt(Path(out_dir) / "velocity.csv", vel_flat, delimiter=",")

        # Acceleration CSV
        acc_flat = acc.reshape(acc.shape[0], -1)
        np.savetxt(Path(out_dir) / "acceleration.csv", acc_flat, delimiter=",")

        print(f"Saved kinematics + CSVs to {out_dir}")
# CLI
def main():
    parser = argparse.ArgumentParser(description="Skeleton lab for pitching mechanics")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--video_b", type=str, default=None, help="Second video for alignment comparison")
    parser.add_argument("--config", type=str, default=None, help="Alignment JSON for video + video_b")
    parser.add_argument("--out_dir", required=True, help="Output directory")

    parser.add_argument("--save_png", action="store_true", help="Save skeleton frames as PNG")
    parser.add_argument("--save_video", action="store_true", help="Save skeleton video")
    parser.add_argument("--overlay", action="store_true", help="Overlay skeleton on original video")
    parser.add_argument("--side_by_side", action="store_true", help="Side-by-side comparison")
    parser.add_argument("--plot_3d", action="store_true", help="Save 3D skeleton plots")
    parser.add_argument("--compute_kinematics", action="store_true", help="Compute velocity/acceleration")

    args = parser.parse_args()

    process_video(
        video_path=args.video,
        out_dir=args.out_dir,
        save_png=args.save_png,
        save_video=args.save_video,
        overlay=args.overlay,
        side_by_side=args.side_by_side,
        plot_3d_flag=args.plot_3d,
        compute_kinematics_flag=args.compute_kinematics,
        video_b_path=args.video_b,
        config_path=args.config
    )

if __name__ == "__main__":
    main()
