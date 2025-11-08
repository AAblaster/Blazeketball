import cv2
import mediapipe as mp
import numpy as np
import sys

# ------------------------------
# PARAMETERS
# ------------------------------
video_path = sys.argv[1]  # Input video path
output_bvh = "blazepose_hierarchy_visible.bvh"
scale = 100  # Scale keypoints for BVH viewers

# ------------------------------
# 1. BlazePose 3D Keypoints
# ------------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video_path}")

all_keypoints = []
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_world_landmarks:
        frame_points = [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]
    else:
        frame_points = [[np.nan, np.nan, np.nan] for _ in range(33)]
    all_keypoints.append(frame_points)

cap.release()
all_keypoints = np.array(all_keypoints, dtype=np.float32)
print(f"3D keypoints shape: {all_keypoints.shape}")

# Save raw data
np.save("blazepose_3d.npy", all_keypoints)

# ------------------------------
# 2. Map 33 landmarks → 17-joint skeleton
# ------------------------------
mp_to_17 = [
    0,   # nose
    11,  # left shoulder
    12,  # right shoulder
    13,  # left elbow
    14,  # right elbow
    15,  # left wrist
    16,  # right wrist
    23,  # left hip
    24,  # right hip
    25,  # left knee
    26,  # right knee
    27,  # left ankle
    28,  # right ankle
    5,   # left eye
    2,   # right eye
    7,   # left ear
    4    # right ear
]

keypoints_17 = all_keypoints[:, mp_to_17, :] * scale
np.save("blazepose_17.npy", keypoints_17)

# ------------------------------
# 3. Skeleton hierarchy (parent indices)
# ------------------------------
joint_names = [
    "Hips",         # 0
    "Spine",        # 1
    "Neck",         # 2
    "Head",         # 3
    "LeftShoulder", # 4
    "LeftElbow",    # 5
    "LeftWrist",    # 6
    "RightShoulder",# 7
    "RightElbow",   # 8
    "RightWrist",   # 9
    "LeftHip",      # 10
    "LeftKnee",     # 11
    "LeftAnkle",    # 12
    "RightHip",     # 13
    "RightKnee",    # 14
    "RightAnkle",   # 15
    "HeadTop"       # 16 (optional top head)
]

# Map BlazePose 17-joint indices to hierarchy
blaze_to_hier = [
    0,  # Hips -> use left hip midpoint
    1,  # Spine -> nose midpoint (approx)
    2,  # Neck -> nose
    3,  # Head -> nose
    4,  # LeftShoulder
    5,  # LeftElbow
    6,  # LeftWrist
    7,  # RightShoulder
    8,  # RightElbow
    9,  # RightWrist
    10, # LeftHip
    11, # LeftKnee
    12, # LeftAnkle
    13, # RightHip
    14, # RightKnee
    15, # RightAnkle
    3   # HeadTop -> nose
]

# Parent indices for BVH
parent_idx = [
    -1, 0, 1, 2, 2, 4, 5, 2, 7, 8, 0, 10, 11, 0, 13, 14, 3
]

# Compute offsets relative to parent (first frame)
offsets = np.zeros((17, 3))
for i in range(17):
    if parent_idx[i] == -1:
        offsets[i] = keypoints_17[0, blaze_to_hier[i]]
    else:
        offsets[i] = keypoints_17[0, blaze_to_hier[i]] - keypoints_17[0, blaze_to_hier[parent_idx[i]]]

# ------------------------------
# 4. BVH exporter
# ------------------------------
def write_joint(f, idx):
    f.write(f"JOINT {joint_names[idx]}\n{{\n")
    f.write(f"  OFFSET {offsets[idx][0]:.3f} {offsets[idx][1]:.3f} {offsets[idx][2]:.3f}\n")
    f.write("  CHANNELS 3 Xrotation Yrotation Zrotation\n")
    for i, p in enumerate(parent_idx):
        if p == idx:
            write_joint(f, i)
    f.write("}\n")

with open(output_bvh, "w") as f:
    f.write("HIERARCHY\n")
    # Root
    f.write(f"ROOT {joint_names[0]}\n{{\n")
    f.write(f"  OFFSET {offsets[0][0]:.3f} {offsets[0][1]:.3f} {offsets[0][2]:.3f}\n")
    f.write("  CHANNELS 3 Xrotation Yrotation Zrotation\n")
    for i, p in enumerate(parent_idx):
        if p == 0:
            write_joint(f, i)
    f.write("}\n")
    # Motion
    f.write("MOTION\n")
    f.write(f"Frames: {keypoints_17.shape[0]}\n")
    f.write("Frame Time: 0.033\n")
    # Placeholder rotations
    for _ in range(keypoints_17.shape[0]):
        line = " ".join(["0"] * 3 * 17)
        f.write(line + "\n")

print(f"✅ Hierarchical BVH saved: {output_bvh}")
