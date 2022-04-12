import os
import pickle
import argparse
from tqdm import tqdm
import torch.nn.functional as F

import numpy as np
from PIL import Image
import torch.utils.data as data

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid

from lf_loader import lostandfoundLoader

from eval import eval_net

from utils import train_step, val_step, get_model, save_viz, save_checkpoint

parser = argparse.ArgumentParser(description="Feature Memory for Anomaly Detection")

# basic config
parser.add_argument('--epochs', type=int,  default=2, help=' The number of epochs to train the memory.')
parser.add_argument('--batch_size', type=int,  default=4, help=' The batch size for training, validation and testing.')
parser.add_argument('--learning_rate', type=float,  default=3e-4, help='Learning rates of model.')
parser.add_argument('--size', type=int,  default=128, help='Side length of input image')
parser.add_argument('--height', type=int,  default=128, help='Height of input image')
parser.add_argument('--width', type=int,  default=128, help='Width of input image')
parser.add_argument('--train_perc', type=float,  default=.9, help='Proportion of samples to use in training set')
parser.add_argument('--data_path', type=str,  default="/scratch/ssd002/datasets/lostandfound", help='The root directory of the dataset.')
parser.add_argument('--ckpt_path', type=str,  default="ckpt/run_1.pth", help='The file to save model checkpoints.')
parser.add_argument('--best_ckpt_path', type=str,  default="ckpt/best_run_1.pth", help='The file to save best model checkpoint.')
parser.add_argument('--sample_path', type=str,  default="samples", help='The file to save best model checkpoint.')


args = parser.parse_args()

# Global Variables 
IMG_SIZE = (args.height, args.width) #H, W
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CURRENT_EPOCH = 0

LF_MAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
}

def main(): 

    # Prepare Dataset and Dataloader
    dataset = lostandfoundLoader(args.data_path, is_transform=True, augmentations=None)

    train_size = int(len(dataset) * args.train_perc)
    val_size = len(dataset) - train_size 
    train_dataset, val_dataset =  torch.utils.data.random_split(dataset, [train_size, val_size]) 

    train_dataloader = data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    val_dataloader = data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)

    model = get_model(pretrained=True)

    # Loss and Optimizer
    criterion = CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Save Initial checkpoint to be subsquently restored from 
    save_checkpoint(model, opt, epoch=CURRENT_EPOCH, path=args.ckpt_path)

    train_loss_list = []
    val_loss_list = []
    max_val_loss = 1e10
    while True:
        # Load checkpoint
        ckpt = torch.load(args.ckpt_path) 

        epoch = ckpt["epoch"]

        if epoch == args.epochs: 
            break 

        model = get_model(pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.to(DEVICE)

        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        opt.load_state_dict(ckpt["opt"])

        model.train()
        train_loss = train_step(model, opt, criterion, train_dataloader, epoch, DEVICE)
        train_loss_list.append(train_loss)

        model.eval()
        val_loss =  eval_net(model, val_dataloader, DEVICE)
        val_loss_list.append(val_loss)


        with open("train_loss.txt", "a") as myfile:
            myfile.write(f"{str(epoch)}\t{str(train_loss)}\n")

        with open("val_loss.txt", "a") as myfile:
            myfile.write(f"{str(epoch)}\t{str(val_loss)}\n")

        if val_loss < max_val_loss: 
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                },
            args.best_ckpt_path)

        save_checkpoint(model, opt, epoch + 1, args.ckpt_path)
        model.cpu()


    f, axarr = plt.subplots(1, 2, figsize=(20,20))
    axarr[0].plot(train_loss_list)
    axarr[0].title.set_text("Train Loss") 
    axarr[1].plot(val_loss_list)
    axarr[1].title.set_text("Validation Loss") 

    fig_path = f"{args.sample_path}/loss_figure.jpg"
    f.savefig(fig_path)

if __name__ == "__main__":
    main()