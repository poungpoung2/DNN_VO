class Config:
    """Configuration class for visual odometry dataset and model parameters"""
    def __init__(self, config_file=None):
        # Default configuration
        self.clip_size = 3  
        self.image_height = 192
        self.image_width = 640
        self.batch_size = 4
        self.num_workers = 4
        self.normalize_means = [0.485, 0.456, 0.406]
        self.normalize_stds = [0.229, 0.224, 0.225]