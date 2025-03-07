class Config():

    def __init__(self):
        self.data_dir = "/home/undergrad3203/Downloads/data/V1_01_easy" # directory to store data
        self.batch_size = 4 # size of batch
        self.val_split = 0.1 # percentage to use as validation set
        self.num_frames = 3 # number of frames
        self.lr = 1e-5 # learning rate
        self.epoch = 100
        self.epoch_init = 1
        self.pretrained = None # load pretrained weights
        self.checkpoint_path = "checkpoints/Exp4" # path to save checkpoint
        self.checkpoint = None


        # tiny  - patch_size=16, embed_dim=192, depth=12, num_heads=3
        # small - patch_size=16, embed_dim=384, depth=12, num_heads=6
        # base  - patch_size=16, embed_dim=768, depth=12, num_heads=12
        self.dim = 384
        self.image_size = (194, 640)
        self.patch_size = 16
        self.num_classes = 6 * (self.num_frames - 1)
        self.depth = 12
        self.num_heads = 6
        self.dim_head = 64
        self.attn_dropout = 0.1
        self.ff_dropout = 0.1

