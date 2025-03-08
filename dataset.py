import torch
<<<<<<< HEAD
=======
import os
>>>>>>> 8f68463ff953d2558688b362c979be2af3d09cf3
from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import v2
import pandas as pd
from PIL import Image
import numpy as np
from utils import get_relative_pose


class VODataset(Dataset):
    def __init__(self, config, base_dir, cam_num, transform=None):
        # Initialize data, download, etc.
<<<<<<< HEAD
        self.image_dir = base_dir / f"cam{cam_num}"
        self.pose_dir = base_dir / "poses.csv"


        if transform is None:
            self.transforms = v2.Compose(
                [
                    v2.Resize(size=config.image_size),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transforms = transform

        self.num_frames = config.num_frames
=======
        image_dir = os.path.join(Path(config.data_dir), "cam0")
        pose_dir = os.path.join(Path(config.data_dir), "poses.csv")
        self.image_dir = Path(image_dir)

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
>>>>>>> 8f68463ff953d2558688b362c979be2af3d09cf3
        self.clips = []
        for i in range(len(list(self.image_dir.glob("*.jpg"))) - self.num_frames + 1):
            self.clips.append((i, i + self.num_frames))

        df = pd.read_csv(self.pose_dir)

        self.poses = {}

        for _, row in df.iterrows():
            image_key = row["image"]

            pose_params = {
                "position": np.array([row["x"], row["y"], row["z"]]),
                "orientation": np.array([row["roll"], row["pitch"], row["yaw"]]),
            }

            self.poses[image_key] = pose_params

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_indices = self.clips[idx]

        # Load images
        images = []
        for img_idx in range(clip_indices[0], clip_indices[1]):
            img_path = self.image_dir / f"{img_idx:06d}.jpg"
            img = Image.open(img_path).convert("RGB")
            img = self.transforms(img)
            images.append(img)

        # Calculate relative poses
        poses = []
        for i in range(clip_indices[0], clip_indices[1]-1):
            curr_pose = self.poses[i]
            next_pose = self.poses[i+1]

            rel_pose = get_relative_pose(curr_pose, next_pose)
            poses.append(rel_pose)

        # Stack tensors
        image_tensor = torch.stack(images, dim=0)
        pose_tensor = torch.tensor(np.array(poses), dtype=torch.float32)

        return image_tensor, pose_tensor
