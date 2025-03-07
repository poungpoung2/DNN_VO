import torch
import numpy as np
import os
from functools import partial
import torch.nn as nn
from model.network import VisionTransformer



def get_model(args, model_params):

    model = VisionTransformer(img_size=model_params["image_size"],
                              num_classes=model_params["num_classes"],
                              patch_size=model_params["patch_size"],
                              embed_dim=model_params["dim"],
                              depth=model_params["depth"],
                              num_heads=model_params["heads"],
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(nn.LayerNorm, eps=1e-6),
                              drop_rate=0.,
                              attn_drop_rate=model_params["attn_dropout"],
                              drop_path_rate=model_params["ff_dropout"],
                              num_frames=model_params["num_frames"])

    # TODO: load checkpoint
    return model, args

