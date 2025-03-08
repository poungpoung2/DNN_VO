import torch
import os
from torch.utils.data import Dataset
from pathlib import Path
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np
from utils import get_relative_pose


class VODataset(Dataset):
    def __init__(self, config, transform=None):
        # Initialize data, download, etc.
        image_dir = []
        pose_dir = []
        self.image_dir = []
        for path in config.data_dir:
            image_dir.append(Path(os.path.join(path, "cam0")))
            pose_dir.append(os.path.join(Path(path), "poses.csv"))
            
        self.image_dir = image_dir

        if transform is None:
                # preprocessing operation
            self.transforms = transforms.Compose([
                transforms.Resize((config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.34721234, 0.36705238, 0.36066107],
                    std=[0.30737526, 0.31515116, 0.32020183]),
            ])
        else:
            self.transforms = transform

        self.clip_size = config.num_frames
        self.clips = []
        for dir in self.image_dir:
            row = []
            for i in range(len(list(dir.glob("*.jpg"))) - self.clip_size + 1):
                row.append((i, i + self.clip_size))
            self.clips.append(row)

        self.flattened_clips = [element for sublist in self.clips for element in sublist]
        
        self.poses = []
        for dir in pose_dir:
            df = pd.read_csv(dir)

            dict = {}

            for _, row in df.iterrows():
                image_key = row["image"]

                pose_params = {
                    "position": np.array([row["x"], row["y"], row["z"]]),
                    "orientation": np.array([row["roll"], row["pitch"], row["yaw"]]),
                }

                dict[image_key] = pose_params
            self.poses.append(dict)

    def __len__(self):
        return len(self.flattened_clips)

    def __getitem__(self, idx):
        row_idx = 0
        column_idx = 0
        for row in self.clips:
            if idx < len(row):
                column_idx = idx
                break
            idx -= len(row)
            row_idx += 1

        clip_indices = self.clips[row_idx][column_idx]

        # Load images
        images = []
        for img_idx in range(clip_indices[0], clip_indices[1]):
            img_path = self.image_dir[row_idx] / f"{img_idx:06d}.jpg"
            img = Image.open(img_path).convert("RGB")
            img = self.transforms(img)
            images.append(img)

        # Calculate relative poses
        poses = []
        for i in range(clip_indices[0], clip_indices[1]-1):
            curr_pose = self.poses[row_idx][i]
            next_pose = self.poses[row_idx][i+1]

            rel_pose = get_relative_pose(curr_pose, next_pose)
            poses.append(rel_pose)

        # Stack tensors
        image_tensor = torch.stack(images, dim=0)
        pose_tensor = torch.tensor(np.array(poses), dtype=torch.float32)
        pose_tensor = pose_tensor.reshape((self.clip_size - 1)*6)

        return image_tensor, pose_tensor
