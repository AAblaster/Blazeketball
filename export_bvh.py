import numpy as np

keypoints_3d = np.load("keypoints_3d.npy")  # (frames, 17, 3)

def export_bvh(filename, data):
    # Very basic BVH exporter for stick skeleton
    with open(filename, "w") as f:
        f.write(f"HIERARCHY\nROOT Hips\n")
        # Add simple hierarchy and motion frames...
        f.write(f"MOTION\nFrames: {data.shape[0]}\n")
        f.write("Frame Time: 0.033\n")  # ~30 FPS
        for frame in data:
            line = " ".join(str(coord) for joint in frame for coord in joint)
            f.write(line + "\n")

export_bvh("shot_motion.bvh", keypoints_3d)
print("✅ BVH exported: shot_motion.bvh")
