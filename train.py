import os
import torch
import torch.optim as optim
from tqdm import tqdm
from get_model import get_model
from torchvision import transforms
from dataset import VODataset
from config import Config
from torch.utils.data import random_split
import pickle
import json

IMG_PATH = "../Data/..."
POSE_PATH = "../Data/..."

torch.manual_seed(2023)

def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0

    with tqdm(val_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            estimate_pose = model(images.float())

            loss = compute_loss(estimate_pose, gt, criterion, args)

            epoch_loss += loss.item()

    
    return epoch_loss / len(val_loader)

def train_epoch(model, train_loader, criterion, optimizer, epoch, args):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1
    with tqdm(train_loader, unit="batch") as tepoch:
        for images, gt in tepoch:
            if torch.cuda.is_available():
                images, gt = images.cudo(), gt.cuda()

            # predict pose
            estimated_pose = model(images.float())
            loss = compute_loss(estimated_pose, gt, criterion, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iter += 1
    
    return epoch_loss / len(train_loader)

def train(model, train_loarder, val_loader, criterion, optimizer, args):
    epochs = args["epoch"]
    checkpoint_path = args["checkpoint_path"]

    for epoch in range(len(epochs)):
        model.train()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, epoch, args)

        # validate model
        if val_loader:
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion, args)

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


def get_optimizer(params, args):
    method = args["optimizer"]

    if method == "Adam":
        optimizer = optim.Adam(params, lr=args["lr"])
    
    return optimizer

def compute_loss(y_hat, y, criterion, args):

    if args["weighted_loss"] == None:
        loss = criterion(y_hat, y.float())
    else:
        loss = criterion(y_hat, y.float())

    return loss

if __name__ == "__main__":

    # set hyperparameters and configuration
    args = {
        "data_dir": "data",
        "bsize": 4,  # batch size
        "val_split": 0.1,  # percentage to use as validation data
        "window_size": 3,  # number of frames in window
        "overlap": 1,  # number of frames overlapped between windows
        "optimizer": "Adam",  # optimizer [Adam, SGD, Adagrad, RAdam]
        "lr": 1e-5,  # learning rate
        "momentum": 0.9,  # SGD momentum
        "weight_decay": 1e-4,  # SGD momentum
        "epoch": 100,  # train iters each timestep
    	"weighted_loss": None,  # float to weight angles in loss function
      	"pretrained_ViT": True,  # load weights from pre-trained ViT
        "checkpoint_path": "checkpoints/Exp4",  # path to save checkpoint
        "checkpoint": None,  # checkpoint
    }

    # tiny  - patch_size=16, embed_dim=192, depth=12, num_heads=3
    # small - patch_size=16, embed_dim=384, depth=12, num_heads=6
    # base  - patch_size=16, embed_dim=768, depth=12, num_heads=12
    model_params = {
        "dim": 384,
        "image_size": (192, 640),  #(192, 640), (640, 640)
        "patch_size": 16,
        "attention_type": 'divided_space_time',  # ['divided_space_time', 'space_only','joint_space_time', 'time_only']
        "num_frames": args["window_size"],
        "num_classes": 6 * (args["window_size"] - 1),  # 6 DoF for each frame
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "time_only": False,
    }
    args["model_params"] = model_params

    dataset_cfg = Config()

    # create checkpoints folder
    if not os.path.exists(args["checkpoint_path"]):
        os.makedirs(args["checkpoint_path"])

    with open(os.path.join(args["checkpoint_path"], 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args["checkpoint_path"], 'args.txt'), 'w') as f:
        f.write(json.dumps(args))

    # train and val dataloader
    print("Using CUDA: ", torch.cuda.is_available())
    print("Loading data...")

    # preprocessing operation
    preprocess = transforms.Compose([
        transforms.Resize((model_params["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.34721234, 0.36705238, 0.36066107],
            std=[0.30737526, 0.31515116, 0.32020183]),
    ])

    dataset = VODataset(dataset_cfg, image_dir=IMG_PATH, pose_path=POSE_PATH)
    nb_val = round(args["val_split"] * len(dataset))

    train_data, val_data = random_split(dataset, [len(dataset) - nb_val, nb_val]) #generator=torch.Generator().manual_seed(2))
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args["bsize"],
                                               shuffle=True,
                                               )
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1,
                                             shuffle=False,
                                             )

    # build and load model
    print("Building model...")
    model, args = get_model(args, model_params)

    # loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = get_optimizer(model.parameters(), args)

    # train network
    print(20*"--" +  " Training " + 20*"--")
    train(model, train_loader, val_loader, criterion, optimizer, args)

