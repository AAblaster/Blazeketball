"""
blazepose_full_pipeline.py

Usage:
  python blazepose_full_pipeline.py input_video.mp4

Outputs:
  - blazepose_3d.npy        (frames, 33, 3) raw world landmarks
  - blazepose_17.npy       (frames, 17, 3) simplified joints (scaled)
  - blazepose_anim.bvh     hierarchical BVH with root translation + rotations
  - a quick matplotlib window will animate the skeleton for verification
"""

import sys
import os
from pathlib import Path
import math
import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Parameters
# -----------------------------
SCALE = 100.0          # scale meters -> viewer units
FPS = 30.0             # frame time for BVH (Frame Time = 1/FPS)
MIN_CONF = 0.5         # detection confidence
MODEL_COMPLEXITY = 2   # BlazePose quality
OUT_BVH = "blazepose_anim.bvh"

# -----------------------------
# Joint mapping & hierarchy
# We define 17 joints and their parent indices.
# Index order used in outputs:
# 0 Hips (root), 1 Spine, 2 Neck, 3 Head,
# 4 LeftShoulder,5 LeftElbow,6 LeftWrist,
# 7 RightShoulder,8 RightElbow,9 RightWrist,
# 10 LeftHip,11 LeftKnee,12 LeftAnkle,
# 13 RightHip,14 RightKnee,15 RightAnkle,
# 16 HeadTop (optional)
# -----------------------------
joint_names = [
    "Hips","Spine","Neck","Head",
    "LeftShoulder","LeftElbow","LeftWrist",
    "RightShoulder","RightElbow","RightWrist",
    "LeftHip","LeftKnee","LeftAnkle",
    "RightHip","RightKnee","RightAnkle",
    "HeadTop"
]
# Parent indices in same order (Hips is root -> -1)
parent_idx = [-1, 0, 1, 2, 2, 4, 5, 2, 7, 8, 0, 10, 11, 0, 13, 14, 3]

# Mapping from BlazePose 33 (MediaPipe) -> our 17 indices (the chosen source indices)
# We will compute certain joints as midpoints when needed (e.g., hips midpoint -> Hips)
# The source indices from BlazePose 33 we use:
# indices from earlier: 0 nose, 5 left eye, 2 right eye, 7 left ear, 4 right ear,
# 11 left shoulder,12 right shoulder,13 left elbow,14 right elbow,15 left wrist,16 right wrist,
# 23 left hip,24 right hip,25 left knee,26 right knee,27 left ankle,28 right ankle
blaze_to_src = {
    "nose": 0,
    "left_eye": 5,
    "right_eye": 2,
    "left_ear": 7,
    "right_ear": 4,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28
}

# Helper to build 17-joint index mapping function (from full 33 landmarks)
def build_17_from_33(frame33):
    """
    frame33: (33,3) array or list of 33 landmarks (x,y,z/world)
    returns (17,3) array in our desired order (in meters)
    """
    f = np.array(frame33, dtype=np.float32)
    # If keypoints are NaN, keep them NaN
    def get(idx):
        return f[idx] if idx is not None and idx < len(f) else np.array([np.nan, np.nan, np.nan])
    left_hip = get(blaze_to_src["left_hip"])
    right_hip = get(blaze_to_src["right_hip"])
    # Hips root = midpoint of left & right hip
    hips = (left_hip + right_hip) / 2.0

    # Spine approx = midpoint between hips and shoulders midpoint
    left_sh = get(blaze_to_src["left_shoulder"])
    right_sh = get(blaze_to_src["right_shoulder"])
    shoulders_mid = (left_sh + right_sh) / 2.0
    spine = (hips + shoulders_mid) / 2.0

    neck = shoulders_mid  # approximate
    head = get(blaze_to_src["nose"])
    headtop = head + np.array([0.0, 0.15, 0.0])  # small offset upwards to form head top

    arr = np.zeros((17, 3), dtype=np.float32)
    arr[0] = hips
    arr[1] = spine
    arr[2] = neck
    arr[3] = head
    arr[4] = left_sh
    arr[5] = get(blaze_to_src["left_elbow"])
    arr[6] = get(blaze_to_src["left_wrist"])
    arr[7] = right_sh
    arr[8] = get(blaze_to_src["right_elbow"])
    arr[9] = get(blaze_to_src["right_wrist"])
    arr[10] = left_hip
    arr[11] = get(blaze_to_src["left_knee"])
    arr[12] = get(blaze_to_src["left_ankle"])
    arr[13] = right_hip
    arr[14] = get(blaze_to_src["right_knee"])
    arr[15] = get(blaze_to_src["right_ankle"])
    arr[16] = headtop
    return arr

# -----------------------------
# Step A: extract 3D BlazePose landmarks (pose_world_landmarks)
# -----------------------------
def extract_blazepose_3d(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=MODEL_COMPLEXITY,
                        enable_segmentation=False,
                        min_detection_confidence=MIN_CONF,
                        min_tracking_confidence=MIN_CONF)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    print("Extracting frames with BlazePose (this may take a while)...")
    pbar = tqdm(total=frame_count) if frame_count else None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_world_landmarks:
            # pose_world_landmarks are real-world coords in meters
            pts = [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]
            frames.append(np.array(pts, dtype=np.float32))
        else:
            # fill with NaNs for consistent shape
            frames.append(np.full((33, 3), np.nan, dtype=np.float32))
        if pbar:
            pbar.update(1)
    if pbar:
        pbar.close()
    cap.release()
    frames = np.stack(frames, axis=0)  # (frames, 33, 3)
    print("Extracted:", frames.shape)
    return frames

# -----------------------------
# Step B: convert to 17-joint skeleton (and scale)
# -----------------------------
def convert_to_17(all33):
    n_frames = all33.shape[0]
    out = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for i in range(n_frames):
        out[i] = build_17_from_33(all33[i])
    return out

# -----------------------------
# Rotation helper: compute rotation that rotates v_from -> v_to
# Returns scipy Rotation object; handles degenerate cases
# -----------------------------
def rotation_from_vectors(v_from, v_to):
    # both are (3,) arrays; may contain NaNs
    if np.any(np.isnan(v_from)) or np.any(np.isnan(v_to)):
        return R.identity()
    a = v_from / (np.linalg.norm(v_from) + 1e-9)
    b = v_to / (np.linalg.norm(v_to) + 1e-9)
    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    if dot > 0.999999:
        return R.identity()
    if dot < -0.999999:
        # 180-degree rotation: find orthogonal vector for axis
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / (np.linalg.norm(axis) + 1e-9)
        return R.from_rotvec(axis * math.pi)
    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        return R.identity()
    axis = axis / axis_norm
    angle = math.acos(dot)
    return R.from_rotvec(axis * angle)

# -----------------------------
# Build rest pose offsets (first valid frame) and rest bone vectors
# -----------------------------
def compute_rest_offsets_and_bones(key17):
    # key17: (frames,17,3) scaled units
    # find first frame that is not all NaN
    valid = None
    for i in range(key17.shape[0]):
        if not np.isnan(key17[i]).all():
            valid = i
            break
    if valid is None:
        raise ValueError("No valid frames detected.")
    rest = key17[valid]
    # offsets: child_pos - parent_pos (for BVH offsets)
    offsets = np.zeros((17,3), dtype=np.float32)
    for idx in range(17):
        p = parent_idx[idx]
        if p == -1:
            offsets[idx] = rest[idx]  # root offset in world units
        else:
            offsets[idx] = rest[idx] - rest[p]
    # rest bone vectors for rotation baseline
    rest_bones = {}
    for idx in range(17):
        p = parent_idx[idx]
        if p != -1:
            rest_bones[idx] = rest[idx] - rest[p]
    return valid, offsets, rest_bones

# -----------------------------
# Compute per-frame local rotations (Euler XYZ degrees) for each joint
# -----------------------------
def compute_rotations_per_frame(key17, rest_bones):
    n_frames = key17.shape[0]
    # rotations: shape (n_frames, 17, 3) (Euler XYZ degrees)
    rots = np.zeros((n_frames, 17, 3), dtype=np.float32)
    for f in range(n_frames):
        frame = key17[f]
        for idx in range(17):
            p = parent_idx[idx]
            if p == -1:
                # root rotation: we can set to identity (0,0,0),
                # and use root translation instead for motion
                rots[f, idx] = np.array([0.0, 0.0, 0.0])
            else:
                # compute rotation from rest_bones[idx] -> current bone vector
                if idx not in rest_bones or np.any(np.isnan(rest_bones[idx])):
                    rots[f, idx] = np.array([0.0,0.0,0.0])
                    continue
                cur_parent = frame[p]
                cur_child = frame[idx]
                if np.any(np.isnan(cur_parent)) or np.any(np.isnan(cur_child)):
                    rots[f, idx] = np.array([0.0,0.0,0.0])
                    continue
                v_from = rest_bones[idx]
                v_to = cur_child - cur_parent
                # compute rotation that aligns v_from -> v_to
                Robj = rotation_from_vectors(v_from, v_to)
                # convert to euler degrees XYZ
                try:
                    euler = Robj.as_euler('xyz', degrees=True)
                except Exception:
                    euler = np.array([0.0,0.0,0.0])
                rots[f, idx] = euler
    return rots

# -----------------------------
# BVH writer (root translation + rotations for each joint)
# We will write channels in order:
# root: Xposition Yposition Zposition
# then for each joint (excluding root): Xrotation Yrotation Zrotation
# Motion lines: one line per frame: (3 values root pos) + 3*(n_joints-1) rotation values
# -----------------------------
def write_bvh(filename, offsets, rotations, key17, frame_time=1.0/FPS):
    n_frames = rotations.shape[0]
    n_joints = len(joint_names)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("HIERARCHY\n")
        # Write root
        f.write(f"ROOT {joint_names[0]}\n{{\n")
        f.write(f"  OFFSET {offsets[0][0]:.6f} {offsets[0][1]:.6f} {offsets[0][2]:.6f}\n")
        f.write("  CHANNELS 3 Xposition Yposition Zposition\n")
        # recursively write children
        def write_children(idx, indent="  "):
            for j, p in enumerate(parent_idx):
                if p == idx:
                    f.write(indent + f"JOINT {joint_names[j]}\n")
                    f.write(indent + "{\n")
                    f.write(indent + f"  OFFSET {offsets[j][0]:.6f} {offsets[j][1]:.6f} {offsets[j][2]:.6f}\n")
                    f.write(indent + "  CHANNELS 3 Xrotation Yrotation Zrotation\n")
                    write_children(j, indent + "  ")
                    f.write(indent + "}\n")
        write_children(0, "  ")
        f.write("}\n")
        # Motion
        f.write("MOTION\n")
        f.write(f"Frames: {n_frames}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")
        for f_idx in range(n_frames):
            # root position = key17[f_idx, root] (we want root pos in world coords)
            root_pos = key17[f_idx, 0]  # already scaled
            # Compose line: root pos then rotations for each joint in index order (excluding root)
            line_vals = [root_pos[0], root_pos[1], root_pos[2]]
            for j in range(1, n_joints):
                # use rotations array (XYZ degrees)
                r = rotations[f_idx, j]
                # BVH typically expects Z-up vs Y-up differences; this is a simple approach.
                line_vals.extend([r[0], r[1], r[2]])
            f.write(" ".join(f"{v:.6f}" for v in line_vals) + "\n")
    print(f"BVH written: {filename}")

# -----------------------------
# Quick Matplotlib visualizer for keypoints_17 animation
# -----------------------------
def visualize_keypoints17(key17):
    # key17 shape: (frames, 17, 3)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-200, 200)
    ax.set_ylim(-200, 200)
    ax.set_zlim(-200, 200)
    scat = ax.scatter([], [], [], s=20)

    def update(i):
        pts = key17[i]
        xs = pts[:,0]; ys = pts[:,1]; zs = pts[:,2]
        scat._offsets3d = (xs, ys, zs)
        ax.set_title(f"Frame {i}")
        return scat,

    ani = FuncAnimation(fig, update, frames=key17.shape[0], interval=1000/FPS, blit=False)
    plt.show()

# -----------------------------
# MAIN
# -----------------------------
def main(video_path):
    print("STEP 1: Extracting BlazePose 3D...")
    all33 = extract_blazepose_3d(video_path)  # (frames,33,3)
    np.save("blazepose_3d.npy", all33)

    print("STEP 2: Convert to 17-joint skeleton and scale...")
    key17 = convert_to_17(all33) * SCALE
    np.save("blazepose_17.npy", key17)
    print("saved blazepose_17.npy:", key17.shape)

    print("STEP 3: Compute rest offsets and bone vectors...")
    rest_frame_idx, offsets, rest_bones = compute_rest_offsets_and_bones(key17)
    # offsets are in scaled units; write_bvh expects offsets array
    print("rest frame:", rest_frame_idx)

    print("STEP 4: Compute per-frame rotations...")
    rotations = compute_rotations_per_frame(key17, rest_bones)

    print("STEP 5: Write BVH file...")
    write_bvh(OUT_BVH, offsets, rotations, key17)

    print("STEP 6: Visualize animation (matplotlib preview)...")
    visualize_keypoints17(key17)
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python blazepose_full_pipeline.py your_video.mp4")
        sys.exit(1)
    video = sys.argv[1]
    if not os.path.exists(video):
        print("Video not found:", video)
        sys.exit(1)
    main(video)
