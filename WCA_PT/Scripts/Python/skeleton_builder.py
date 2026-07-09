import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

mp_pose = mp.solutions.pose

POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

def draw_skeleton(image, landmarks):
    #Draws a skeleton using Matplotlib from MediaPipe landmarks.
    plt.figure(figsize=(6, 10))
    plt.imshow(image)
    plt.axis('off')

    # Draw joints
    for lm in landmarks:
        x = lm.x * image.shape[1]
        y = lm.y * image.shape[0]
        plt.scatter(x, y, c='red', s=20)

    # Draw bones
    for c in POSE_CONNECTIONS:
        start = landmarks[c[0]]
        end = landmarks[c[1]]

        x1 = start.x * image.shape[1]
        y1 = start.y * image.shape[0]
        x2 = end.x * image.shape[1]
        y2 = end.y * image.shape[0]

        plt.plot([x1, x2], [y1, y2], c='yellow', linewidth=2)

    plt.show()


def process_video(video_path, frame_step=5):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                draw_skeleton(rgb, results.pose_landmarks.landmark)

        frame_idx += 1

    cap.release()
    pose.close()


if __name__ == "__main__":
    video_path = "Data/Videos/Skenes_CH_000.mp4"
    process_video(video_path, frame_step=3)
