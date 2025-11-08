"""
realtime_blazepose_3d_visualizer.py

- Run: python realtime_blazepose_3d_visualizer.py
- Press Ctrl+C in the terminal to stop and save last_npy files.
"""

import cv2
import mediapipe as mp
import numpy as np
import open3d as o3d
import time
from collections import deque

# ---------------------------
# CONFIG
# ---------------------------
WEBCAM_INDEX = 0
SCALE = 200.0                # scale meter -> viewer units (tweak to taste)
EMA_ALPHA = 0.6              # smoothing: new_smoothed = alpha*new + (1-alpha)*prev
HISTORY_LEN = 3              # small history buffer for simple interpolation
FRAME_TIME = 1.0 / 30.0      # viewer frame time approx
SAVE_ON_EXIT = True

# BlazePose → 17-joint mapping (same as earlier)
MP_TO_17 = [
    0,   # nose (0)
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

JOINT_NAMES_17 = [
    "Nose","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist",
    "LHip","RHip","LKnee","RKnee","LAnkle","RAnkle","LEye","REye","LEar","REar"
]

# Edges for drawing lines (pairs of indices into 17-joint list)
LINES = [
    (7,8),     # hips midpoint shape across hips (LHip-RHip)
    (8,0),     # RHip - Nose (approx spine)
    (7,0),     # LHip - Nose
    (0,1),     # Nose - LShoulder
    (0,2),     # Nose - RShoulder
    (1,3),     # LShoulder - LElbow
    (3,5),     # LElbow - LWrist
    (2,4),     # RShoulder - RElbow
    (4,6),     # RElbow - RWrist
    (7,9),     # LHip - LKnee
    (9,11),    # LKnee - LAnkle
    (8,10),    # RHip - RKnee
    (10,12),   # RKnee - RAnkle
    (0,13),    # Nose - LEye
    (0,14),    # Nose - REye
    (13,15),   # LEye - LEar
    (14,16)    # REye - REar
]

# ---------------------------
# Helpers: interpolation & smoothing
# ---------------------------
def nan_safe_replace(arr, fallback):
    """Replace NaN entries in arr with fallback (both arrays same shape)."""
    out = arr.copy()
    mask = np.isnan(out)
    out[mask] = fallback[mask]
    return out

def simple_interpolate(frame_buffer, idx):
    """
    Given a deque of previous frames (each (17,3)), produce an interpolated current frame:
    - If the newest frame has NaNs, try to linearly interpolate between last valid frames.
    - If no valid frames exist, return zeros.
    """
    # convert to numpy array: (t,17,3)
    hist = np.array(frame_buffer)  # shape (n,17,3)
    # newest = hist[-1]
    newest = hist[-1]
    if not np.isnan(newest).any():
        return newest
    # find last non-nan frame before newest
    for i in range(len(hist)-2, -1, -1):
        if not np.isnan(hist[i]).any():
            prev = hist[i]
            # simple carry-forward fill
            filled = nan_safe_replace(newest, prev)
            return filled
    # nothing valid -> replace NaNs with zeros
    return np.nan_to_num(newest)

def apply_ema(prev_smoothed, new_data, alpha=EMA_ALPHA):
    if prev_smoothed is None:
        return new_data.copy()
    return alpha*new_data + (1.0-alpha)*prev_smoothed

# ---------------------------
# MediaPipe init
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------
# Open3D Visualizer init
# ---------------------------
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="BlazePose 3D Live", width=900, height=700)
# initial dummy points
initial_points = np.zeros((17,3), dtype=np.float64)
lines = np.array(LINES, dtype=np.int32)
colors = np.tile(np.array([[0.9,0.2,0.2]]), (len(lines),1))

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(initial_points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(line_set)

# add coordinate frame
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
vis.add_geometry(frame)

# ---------------------------
# Webcam loop
# ---------------------------
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam index {}".format(WEBCAM_INDEX))

frame_buffer = deque(maxlen=HISTORY_LEN)   # store last few mapped 17-joint frames
smoothed = None                             # store EMA-smoothed frame (17,3)
saved_raw_frames = []                       # keep for optional save

print("Starting webcam -> BlazePose -> smoothing -> 3D display. Press Ctrl+C to quit.")

try:
    while True:
        start_t = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed; retrying...")
            time.sleep(0.01)
            continue

        # Run BlazePose on BGR->RGB frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_world_landmarks:
            lm = results.pose_world_landmarks.landmark
            pts33 = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (33,3)
        else:
            pts33 = np.full((33,3), np.nan, dtype=np.float32)

        # Map 33 -> 17
        try:
            mapped = pts33[MP_TO_17, :]  # shape (17,3)
        except Exception as e:
            mapped = np.full((17,3), np.nan, dtype=np.float32)

        # store to history buffer
        frame_buffer.append(mapped)

        # interpolate NaNs using simple strategy
        interp = simple_interpolate(frame_buffer, idx=0)  # returns (17,3) without many NaNs
        # replace any remaining NaNs with previous smoothed or zeros
        if smoothed is None:
            fallback = np.zeros_like(interp)
        else:
            fallback = smoothed
        interp = nan_safe_replace(interp, fallback)

        # scale and coordinate transform for viewing:
        # - MediaPipe uses right-handed with y up? pose_world_landmarks approximate meters with origin near camera.
        # We multiply by SCALE and flip the z sign so that +z goes out of screen for open3d viewer look.
        
        interp_vis = np.array(interp, dtype=np.float64) * SCALE
        interp_vis[:,1] *= -1.0  # flip y axis
        interp_vis[:,2] *= -1.0  # flip z

        # apply EMA smoothing
        smoothed = apply_ema(smoothed, interp_vis)

        # update Open3D geometry
        line_set.points = o3d.utility.Vector3dVector(smoothed)
        vis.update_geometry(line_set)
        vis.poll_events()
        vis.update_renderer()

        # small optional overlay of camera preview (show in cv2 window)
        preview = frame.copy()
        cv2.putText(preview, "Press Ctrl+C in terminal to quit", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Webcam preview (press q to close cv window)", preview)
        # keep CV window responsive
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        saved_raw_frames.append(mapped)  # store original mapped frames (17,3) in meters

        # pace loop to ~30 FPS
        elapsed = time.time() - start_t
        sleep = FRAME_TIME - elapsed
        if sleep > 0:
            time.sleep(sleep)

except KeyboardInterrupt:
    print("\nInterrupted by user — exiting cleanly...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    # Save recent captured data
    if SAVE_ON_EXIT and len(saved_raw_frames) > 0:
        arr = np.array(saved_raw_frames, dtype=np.float32)   # shape (n_frames,17,3)
        np.save("realtime_blazepose_17_raw.npy", arr)
        print(f"Saved {arr.shape[0]} frames to realtime_blazepose_17_raw.npy")
    pose.close()
    print("Done.")
