import cv2
import mediapipe as mp
import numpy as np
import sys

video_path = sys.argv[1]  # video filename as argument

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(video_path)
all_keypoints = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        frame_points = []
        for lm in results.pose_landmarks.landmark:
            frame_points.append([lm.x, lm.y])
        all_keypoints.append(frame_points)

cap.release()

all_keypoints = np.array(all_keypoints)  # shape: (frames, 33, 2)
np.save("keypoints_2d.npy", all_keypoints)
print("✅ 2D keypoints saved: keypoints_2d.npy")
