# shooting_analyzer_top_half.py
"""
Real-time basketball shooting form analyzer (top-half only, relaxed detection)

Features:
- Webcam BlazePose 3D (pose_world_landmarks)
- NaN-safe temporal interpolation and EMA smoothing
- Shot detection heuristic (relaxed: elbow + wrist velocity)
- Per-shot metrics: release height, elbow angle, shoulder balance
- Human-readable feedback in terminal
- Saves short video clips and raw keypoints
- Overlay skeleton + feedback on webcam preview

Run:
    python shooting_analyzer_top_half.py
Press Ctrl+C in terminal or 'q' in preview window to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import collections
import math
import os
from scipy.signal import savgol_filter
import pyttsx3

# -----------------------------
# CONFIG
# -----------------------------
WEBCAM_INDEX = 0
FPS = 30
FRAME_TIME = 1.0 / FPS
SCALE_FOR_BVH = 100.0        # if exporting BVH
SAVE_CLIP_SECONDS_BEFORE = 1.0
SAVE_CLIP_SECONDS_AFTER = 1.2
CLIP_FPS = FPS

EMA_ALPHA = 0.6
HISTORY_LEN = 15
SAVGOL_WINDOW = 9  # must be odd

# Relaxed thresholds for top-half only
WRIST_VEL_MIN = 0.7
ELBOW_EXTENSION_ANGLE_DELTA_MIN = 4.0
MIN_GATHER_FRAMES = 2
MAX_SHOT_SEQ_FRAMES = 90

EXPORT_BVH_PER_SHOT = False
EXPORT_VIDEO_CLIPS = True
EXPORT_RAW_KEYPOINTS = True
OUTPUT_DIR = "shots_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BlazePose 33 -> 17 joints (simplified)
MP_TO_17 = [
    0, 11, 12, 13, 14, 15, 16, 5, 2, 7, 4, 23, 24, 25, 26, 27, 28
]

JOINT_NAMES_17 = [
    "Nose","LShoulder","RShoulder","LElbow","RElbow","LWrist","RWrist",
    "LEye","REye","LEar","REar","LHip","RHip","LKnee","RKnee","LAnkle","RAnkle","Root"
]

LINES_17 = [
    (7,8),(8,0),(7,0),(0,1),(0,2),(1,3),(3,5),(2,4),(4,6)
]

def speakBall(text_to_speak):
    """
    Initializes pyttsx3 engine, attempts to set a male voice (like Jarvis),
    and speaks the provided text.
    """
    try:
        # Initialize the speech engine
        engine = pyttsx3.init()

        # --- Voice and Rate Customization ---
        
        # Get the list of installed voices
        voices = engine.getProperty('voices')

        # Try to set a male voice. voices[0] is often a male voice on many systems.
        # You may need to change the index (0, 1, 2, etc.) depending on your OS/installed voices.
        if voices:
            # Set the voice property using the ID of the first voice (index 0)
            engine.setProperty('voice', voices[0].id)
            print(f"Using voice: {voices[0].name}")
        else:
            print("Warning: No voices found. Using default voice.")

        # Set a slightly slower rate for a more commanding, deliberate tone (optional)
        # Default is usually 200.
        engine.setProperty('rate', 150)
        
        # --- Speaking the Text ---

        # Queue the text to be spoken
        engine.say(text_to_speak)
        
        # Blocks while all queued commands are processed
        engine.runAndWait()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Make sure you have pyttsx3 installed: pip install pyttsx3")

# -----------------------------
# UTIL
# -----------------------------
def vec(p): return np.array(p, dtype=np.float32)
def length(v): return float(np.linalg.norm(v))
def normalize(v): 
    n = np.linalg.norm(v)
    return v / (n+1e-8)
def angle_between(a,b):
    a_n = normalize(a); b_n = normalize(b)
    cos = np.clip(np.dot(a_n,b_n), -1.0, 1.0)
    return math.degrees(math.acos(cos))

def nan_safe_replace(arr, fallback):
    out = arr.copy()
    mask = np.isnan(out)
    out[mask] = fallback[mask]
    return out

def simple_interpolate_history(buffer):
    hist = np.array(buffer)
    newest = hist[-1]
    if not np.isnan(newest).any():
        return newest
    for i in range(len(hist)-2, -1, -1):
        if not np.isnan(hist[i]).any():
            prev = hist[i]
            return nan_safe_replace(newest, prev)
    return np.nan_to_num(newest)

def apply_ema(prev, new, alpha=EMA_ALPHA):
    if prev is None:
        return new.copy()
    return alpha*new + (1.0-alpha)*prev

# -----------------------------
# SHOT DETECTOR
# -----------------------------
class ShotDetector:
    def __init__(self, wrist_vel_min=WRIST_VEL_MIN,
                 elbow_angle_delta_min=ELBOW_EXTENSION_ANGLE_DELTA_MIN):
        self.wrist_vel_min = wrist_vel_min
        self.elbow_angle_delta_min = elbow_angle_delta_min
        self.reset()

    def reset(self):
        self.state = "idle"
        self.frames = 0
        self.gather_frames = 0
        self.last_elbow_angle = None
        self.seq_start_frame = 0

    def update(self, kp17, prev_kp17, dt):
        self.frames += 1
        if kp17 is None or prev_kp17 is None:
            return None, None

        # Elbow angle (R arm: shoulder idx 2, elbow 4, wrist 6)
        shoulder = kp17[2]; elbow = kp17[4]; wrist_pt = kp17[6]
        vec_upper = shoulder - elbow
        vec_lower = wrist_pt - elbow
        elbow_angle = angle_between(vec_upper, vec_lower)

        # Wrist velocity
        wrist = kp17[6]; prev_wrist = prev_kp17[6]
        wrist_vel = length(wrist - prev_wrist) / (dt+1e-8)

        is_gather = elbow_angle < 145 and wrist_vel < 0.7
        elbow_delta = 0.0 if self.last_elbow_angle is None else self.last_elbow_angle - elbow_angle

        event = None; info = {}

        if self.state == "idle" and is_gather:
            self.state = "gather"
            self.gather_frames = 1
            self.seq_start_frame = self.frames

        elif self.state == "gather":
            if is_gather:
                self.gather_frames += 1
            else:
                self.state = "idle"
                self.gather_frames = 0

            if wrist_vel > self.wrist_vel_min and elbow_delta > self.elbow_angle_delta_min:
                self.state = "idle"
                event = "shot"
                release_height = wrist[1]
                elbow_at_release = elbow_angle
                shoulder_left = kp17[1]; shoulder_right = kp17[2]
                balance = abs(shoulder_left[0] - shoulder_right[0])
                info = {
                    "release_height_m": float(release_height),
                    "elbow_deg": float(elbow_at_release),
                    "wrist_vel": float(wrist_vel),
                    "balance_m": float(balance)
                }
                self.reset()
                return event, info

            if (self.frames - self.seq_start_frame) > MAX_SHOT_SEQ_FRAMES:
                self.reset()

        self.last_elbow_angle = elbow_angle
        return event, info

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    frame_buffer = collections.deque(maxlen=int((SAVE_CLIP_SECONDS_BEFORE+SAVE_CLIP_SECONDS_AFTER)*CLIP_FPS)+10)
    key_buffer = collections.deque(maxlen=HISTORY_LEN)
    smoothed = None
    detector = ShotDetector()
    prev_kp17 = None
    saved_shots = 0
    recorded_clips = []

    last_time = time.time()
    print("Starting analyzer. Press Ctrl+C or 'q' to quit.")

    while True:
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_world_landmarks:
            lm = results.pose_world_landmarks.landmark
            pts33 = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        else:
            pts33 = np.full((33,3), np.nan, dtype=np.float32)

        mapped = pts33[MP_TO_17, :] if pts33.shape[0]>=33 else np.full((17,3), np.nan)

        key_buffer.append(mapped)
        interp = simple_interpolate_history(key_buffer)
        fallback = smoothed if smoothed is not None else np.zeros_like(interp)
        interp = nan_safe_replace(interp, fallback)

        # Savitzky-Golay smoothing
        if len(key_buffer) >= 5:
            hist = np.array(key_buffer)
            for j in range(17):
                for c in range(3):
                    series = np.nan_to_num(hist[:,j,c])
                    try:
                        if len(series) >= SAVGOL_WINDOW:
                            smoothed_series = savgol_filter(series, SAVGOL_WINDOW, 3)
                            interp[j,c] = smoothed_series[-1]
                    except: pass

        smoothed = apply_ema(smoothed, interp)
        curr_kp17 = smoothed.copy()
        now = time.time()
        dt = now - last_time if last_time else FRAME_TIME
        last_time = now

        event, info = detector.update(curr_kp17, prev_kp17 if prev_kp17 is not None else curr_kp17, dt)

        if event=="shot":
            saved_shots +=1
            tstamp = int(time.time())
            print(f"\n=== Shot #{saved_shots} detected ===")
            print(info)
            feedback_lines = []

            if info["release_height_m"] < 1.7:
                feedback_lines.append("Release height low: extend arm and release higher.")
            else:
                feedback_lines.append("Good release height.")

            if info["elbow_deg"] > 100:
                feedback_lines.append("Elbow too open: tuck slightly for better arc.")
            else:
                feedback_lines.append("Elbow position looks good.")

            if info["balance_m"] > 0.15:
                feedback_lines.append("Balance off: shoulders not square.")
            else:
                feedback_lines.append("Balance looks good.")

            print("FEEDBACK:")
            for ln in feedback_lines:
                print(" - "+ln)

        frame_buffer.append(frame.copy())
        prev_kp17 = curr_kp17.copy()

        # Overlay 2D skeleton using MediaPipe landmarks
        preview = frame.copy()
        h,w = preview.shape[:2]
        if results.pose_landmarks:
            for connection in mp.solutions.pose.POSE_CONNECTIONS:
                try:
                    start = results.pose_landmarks.landmark[connection[0]]
                    end   = results.pose_landmarks.landmark[connection[1]]
                    x1,y1 = int(start.x*w), int(start.y*h)
                    x2,y2 = int(end.x*w), int(end.y*h)
                    cv2.line(preview,(x1,y1),(x2,y2),(0,255,0),2)
                except: pass

        status_text = "Idle" if detector.state=="idle" else detector.state.capitalize()
        cv2.putText(preview,f"State: {status_text}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)
        cv2.putText(preview,f"Shots: {saved_shots}",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,255),2)

        if event=="shot":
            y0 = 90
            for ln in feedback_lines:
                cv2.putText(preview,ln,(10,y0),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
                y0+=22

        cv2.imshow("Shooting Analyzer Preview", preview)
        if cv2.waitKey(1)&0xFF==ord('q'): break
        elapsed = time.time()-loop_start
        if FRAME_TIME - elapsed>0: time.sleep(FRAME_TIME-elapsed)

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print("Exiting. Total shots detected:", saved_shots)

if __name__=="__main__":
    main()
