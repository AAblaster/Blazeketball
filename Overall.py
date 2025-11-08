import cv2
import mediapipe as mp
import numpy as np
import torch
from common.model import TemporalModel
import sys
import os

# ------------------------------
# Parameters
# ------------------------------
video_path = sys.argv[1]  # Video file path
output_bvh = "shot_motion.bvh"

# ------------------------------
# 1. Extract 2D keypoints with MediaPipe
# ------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
cap = cv2.VideoCapture(video_path)

keypoints_2d = []

print("🔹 Extracting 2D keypoints from video...")
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
        keypoints_2d.append(frame_points)

cap.release()
keypoints_2d = np.array(keypoints_2d, dtype=np.float32)
print(f"✅ 2D keypoints extracted: {keypoints_2d.shape}")

# ------------------------------
# 2. Map MediaPipe 33 → COCO 17 joints
# ------------------------------
mp_to_coco_map = [
    11, # left shoulder
    12, # right shoulder
    13, # left elbow
    14, # right elbow
    15, # left wrist
    16, # right wrist
    23, # left hip
    24, # right hip
    25, # left knee
    26, # right knee
    27, # left ankle
    28, # right ankle
    0,  # nose
    5,  # left eye
    2,  # right eye
    7,  # left ear
    4,  # right ear
]
keypoints_coco = keypoints_2d[:, mp_to_coco_map, :]
print(f"✅ Converted to COCO 17-joint format: {keypoints_coco.shape}")

# ------------------------------
# 3. Load pretrained VideoPose3D model
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TemporalModel(
    num_joints_in=17,
    in_features=2,
    num_joints_out=17,
    filter_widths=[3, 3, 3, 3, 3]
)

checkpoint_path = "checkpoint/pretrained_h36m_detectron_coco.bin"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"{checkpoint_path} not found. Download the pretrained model!")

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model = model.to(device).eval()

# ------------------------------
# 4. Run 3D lifting
# ------------------------------
keypoints_tensor = torch.from_numpy(keypoints_coco).unsqueeze(0).to(device)
with torch.no_grad():
    keypoints_3d = model(keypoints_tensor).cpu().numpy().squeeze(0)

print(f"✅ 3D keypoints generated: {keypoints_3d.shape}")
np.save("keypoints_3d.npy", keypoints_3d)

# ------------------------------
# 5. Export to BVH (basic stick skeleton)
# ------------------------------
def export_bvh(filename, data):
    n_frames = data.shape[0]
    with open(filename, "w") as f:
        f.write("HIERARCHY\nROOT Hips\n{\n")
        f.write("  OFFSET 0 0 0\n  CHANNELS 3 Xposition Yposition Zposition\n")
        f.write("}\nMOTION\n")
        f.write(f"Frames: {n_frames}\nFrame Time: 0.033\n")
        for frame in data:
            line = " ".join(str(coord) for joint in frame for coord in joint)
            f.write(line + "\n")

export_bvh(output_bvh, keypoints_3d)
print(f"✅ BVH exported: {output_bvh}")
