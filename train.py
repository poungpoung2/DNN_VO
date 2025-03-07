import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import VODataset
from model.network import VisionTransformer
from torch.utils.data import random_split
from config import Config
from functools import partial


torch.manual_seed(2023)

def val_epoch(model, val_loader, criterion):
    epoch_loss = 0

    with tqdm(val_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            estimate_pose = model(images.float())
            loss = criterion(estimate_pose, gt.float())

            epoch_loss += loss.item()

    
    return epoch_loss / len(val_loader)

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            # predict pose
            estimate_pose = model(images.float())
            loss = criterion(estimate_pose, gt.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iter += 1
    
    return epoch_loss / len(train_loader)

def train(model, train_loarder, val_loader, criterion, optimizer, cfg):
    epochs = cfg.epoch
    checkpoint_path = cfg.checkpoint_path
    epoch_init = cfg.epoch_init

    for epoch in range(epoch_init-1, epochs):
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch)

        # validate model
        if val_loader:
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion)

            print(f"Epoch: {epoch} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} \n")

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

        if not epoch % 20:
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_e{}.pth".format(epoch)))
        
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))

    return


if __name__ == "__main__":

    config = Config()

    # create checkpoints folder
    if not os.path.exists(config.checkpoint_path):
        os.makedirs(config.checkpoint_path)

    # train and val dataloader
    print("Using CUDA: ", torch.cuda.is_available())
    print("Loading data...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = VODataset(config=config)
    nb_val = round(config.val_split * len(dataset))

    train_data, val_data = random_split(dataset, [len(dataset) - nb_val, nb_val]) 
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1,
                                             shuffle=False,
                                             )

    # build and load model
    print("Building model...")
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
    model = model.to(device)

    # loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if config.pretrained is not None:
        state_dict = torch.load(config.pretrained)
        model.load_state_dict(state_dict["model_state_dict"])

    elif config.checkpoint is not None:
        checkpoint = torch.load(os.path.join(config.checkpoint_path, config.checkpoint))
        config.epoch_init = checkpoint["epoch"]
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # train network
    print(20*"--" +  " Training " + 20*"--")
    train(model, train_loader, val_loader, criterion, optimizer, config)

