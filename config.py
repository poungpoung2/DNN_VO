from pathlib import Path
import pickle


class Config:
    def __init__(self, config_file=None):
        self.device = None
        # Model settings
        self.clip_size = 3
        self.img_height = 192
        self.img_width = 640
        self.batch_size = 4
        self.num_workers = 4
        self.data_dir = Path("data")

        self.config_path = Path("config.pkl")

        # Training
        self.lr = 1e-5

        # Checkpoint
        self.checkpoint_dir = Path("checkpoint")
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.best_loss = float("inf")
        self.best_loss_epoch = 0
        self.global_epoch = 0

        if config_file:
            self.config_path = Path(config_file)
            if self.config_path.exists():
                self.load_config()

    def save_config(self):
        with self.config_path as file:
            pickle.dump(self, file)

    def load_config(self):
        if self.config_path.exists():
            with self.config_path.open("rb") as file:
                loaded_config = pickle.load(file)
                self.__dict__.update(loaded_config.__dict__)
        else:
            print(f"Config file {self.config_path} not found.")
        return self
