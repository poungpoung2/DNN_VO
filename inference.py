import torch
from dataset import VODataset
from model.network import VisionTransformer
from model.deepvo import DeepVO
from functools import partial
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
from config import Config
from deepvo_config import DeepVOConfig
import pandas as pd
from utils import pose_to_trans, rot_to_euler
from pathlib import Path

# checkpoint_path = "checkpoints/Exp2"
# checkpoint_name = "best"
# checkpoint_path = "/home/undergrad3203/Downloads"
# checkpoint_name = "checkpoint_model2_exp19"
checkpoint_path = "checkpoints/DEEPVO"
checkpoint_name = "best"
dataset_path = "/home/undergrad3203/Downloads/data/V1_01_easy"
# initial_pose_dict = {
#     "position": np.array([-0.982487, 0.46278, 1.4401]),
#     "orientation": np.array([3.128505, -0.034012, 0.035006])
# }
initial_pose_dict = {
    "position": np.array([0.7866, 2.176941, 1.062022]),
    "orientation": np.array([0.228362, 0.046038, -0.013076])
}
# initial_pose_dict = {
#     "position": np.array([0.993828, 2.276816, 1.339974]),
#     "orientation": np.array([0.146015, 0.019777, -0.012613])
# }

device = "cuda" if torch.cuda.is_available() else "cpu"

config = DeepVOConfig()
config.checkpoint_path = checkpoint_path

# Build and load model
# model = VisionTransformer(
#     img_size=config.image_size,
#     num_classes=config.num_classes,
#     patch_size=config.patch_size,
#     embed_dim=config.dim,
#     depth=config.depth,
#     num_heads=config.num_heads,
#     mlp_ratio=4,
#     qkv_bias=True,
#     norm_layer=partial(nn.LayerNorm, eps=1e-6),
#     drop_rate=0.,
#     attn_drop_rate=config.attn_dropout,
#     drop_path_rate=config.ff_dropout,
#     num_frames=config.num_frames
# )

model = DeepVO(image_size=config.image_size,
                   num_classes=config.num_classes,
                   input_channels=config.num_frames*3,
                   batch_size=config.batch_size,
                   hidden_size=config.hidden_size)

checkpoint_file = os.path.join(config.checkpoint_path, f"{checkpoint_name}.pth")
checkpoint = torch.load(checkpoint_file, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()  # Set once before the loop

print("LOADING DATASET...")
dataset = VODataset(config, Path(dataset_path), cam_num=0)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize a list for predicted poses.
# Convert the initial pose to a (6,) vector.
initial_pose_vec = np.concatenate((initial_pose_dict["position"], initial_pose_dict["orientation"]))
pred_poses = [initial_pose_vec]

# Process the data
with tqdm(test_loader, unit="batch") as batchs:
    skip = 0
    take = 0
    for images, gt in batchs:
        
        # take += 1
        # if take < 149:
        #     continue
        # if skip == config.num_frames-2:
        #     skip = 0
        #     continue
        # skip += 1
        # if skip == 300:
        #     break
        images = images.to(device).float()
        with torch.no_grad():
            # Forward pass: output shape should be (config.num_frames-1)*6
            pred_pose = model(images)
            # Reshape to (config.num_frames-1, 6) and convert to NumPy
            pred_pose = pred_pose.reshape(config.num_frames - 1, 6).cpu().numpy()
            
            # For each relative pose predicted, update the trajectory.
            for i in range(config.num_frames - 1):
                # Extract the relative pose as a (6,) vector.
                rel_pose_vec = pred_pose[i]
                rel_pose = {
                    "position": rel_pose_vec[3:],
                    "orientation": rel_pose_vec[:3]
                }
                # Get the current pose (last predicted pose).
                current_pose_vec = pred_poses[-1]
                current_pose = {
                    "position": current_pose_vec[:3],
                    "orientation": current_pose_vec[3:]
                }
                # Convert current and relative poses to transformation matrices.
                T_current = pose_to_trans(current_pose)
                T_rel = pose_to_trans(rel_pose)
                # Compose the transformations.
                T_new = T_current @ T_rel
                new_position = T_new[:3, 3]
                new_orientation = rot_to_euler(T_new[:3, :3])
                # Create the new pose vector and append it.
                new_pose_vec = np.concatenate((new_position, new_orientation))
                pred_poses.append(new_pose_vec)

# Convert the list of poses to a NumPy array.
pred_poses_array = np.vstack(pred_poses)

# Save to CSV.
df = pd.DataFrame(pred_poses_array, columns=["x", "y", "z", "roll", "pitch", "yaw"])
output_csv = os.path.join(dataset_path, "deepvo.csv")
df.to_csv(output_csv, index=False)
print(f"Output saved to {output_csv}")
