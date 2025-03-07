import glob
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


class KITTI(torch.utils.data.Dataset):

    def __init__(self,
                 data_path=r"data/sequences_jpg",
                 gt_path=r"data/poses",
                 camera_id="2",
                 sequences=["00", "02", "08", "09"],
                 window_size=3,
                 overlap=1,
                 read_poses=True,
                 transform=None,
                 ):
        
        self.data_path = data_path
        self.gt_path = gt_path
        self.camera_id = camera_id
        self.frame_id = 0
        self.read_poses = read_poses
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform

    # NEED TO FINISH DATASET