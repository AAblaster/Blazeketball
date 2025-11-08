import torch
import numpy as np
from common.model import TemporalModel

# Load 2D COCO keypoints
keypoints_2d = np.load("keypoints_2d_coco.npy")  # (frames, 17, 2)

# Convert to tensor: shape (1, frames, 17, 2)
keypoints_tensor = torch.from_numpy(keypoints_2d.astype(np.float32)).unsqueeze(0)

# Load pretrained model
model = TemporalModel(
    num_joints_in=17,
    in_features=2,
    num_joints_out=17,
    filter_widths=[3,3,3,3,3]
)
checkpoint = torch.load("checkpoint/pretrained_h36m_detectron_coco.bin", map_location="cuda")
model.load_state_dict(checkpoint['model'], strict=False)
model = model.cuda().eval()

# Run inference
with torch.no_grad():
    keypoints_3d = model(keypoints_tensor.cuda()).cpu().numpy().squeeze(0)

np.save("keypoints_3d.npy", keypoints_3d)
print("✅ 3D keypoints saved: keypoints_3d.npy")
