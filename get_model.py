import torch
import numpy as np
import os
from functools import partial
import torch.nn as nn
from model.network import VisionTransformer



def get_model(config):

    model = VisionTransformer(img_size=config.image_size,
                              num_classes=config.num_classes,
                              patch_size=config.patch_size,
                              embed_dim=config.dim,
                              depth=config.depth,
                              num_heads=config.num_heads,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              drop_rate=0.,
                              attn_drop_rate=config.attn_dropout,
                              drop_path_rate=config.ff_dropout,
                              num_frames=config.num_frames)

    # TODO: load model and checkpoint
    if config.pretrained:
        return model

    if config.load_checkpoint is not None:
        return model

    return model

