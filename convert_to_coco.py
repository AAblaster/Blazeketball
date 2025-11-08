import numpy as np

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

keypoints_2d = np.load("keypoints_2d.npy")  # shape: (frames, 33, 2)
keypoints_coco = keypoints_2d[:, mp_to_coco_map, :]
np.save("keypoints_2d_coco.npy", keypoints_coco)
print("✅ Converted to COCO 17-joint format")
