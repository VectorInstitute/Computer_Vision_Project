import os
import argparse

import numpy as np

import random

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader


from dataset import SpaceNet_Dataset
from metrics import iou

from utils import save_viz,  save_checkpoint, get_model

from training import get_label_dist, train_fn, val_fn, test_fn

parser = argparse.ArgumentParser(description="Feature Memory for Anomaly Detection")

# basic config
parser.add_argument('--model', type=str, help='Architecture variation for experiments')
parser.add_argument('--epochs', type=int,  default=25, help=' The number of epochs to train the memory.')
parser.add_argument('--batch_size', type=int,  default=8, help=' The batch size for training, validation and testing.')
parser.add_argument('--learning_rate', type=float,  default=2e-4, help='Learning rates of memory units.')
parser.add_argument('--size', type=int,  default=384, help='Side length of input image')
parser.add_argument('--train_perc', type=float,  default=.8, help='The proportion of train samples used for validation.')
parser.add_argument('--val_perc', type=float,  default=.1, help='The proportion of train samples used for validation.')
parser.add_argument('--data_path', type=str,  default="/scratch/ssd002/datasets/cv_project/spacenet", help='The root directory of the dataset.')

args = parser.parse_args()

CITY_LIST = ["Vegas", "Paris", "Shanghai",  "Khartoum"]

TRAIN_IMG_PATHS = [os.path.join(args.data_path, f"AOI_{i+2}_{city}_Train_processed", "RGB-PanSharpen") for i, city in enumerate(CITY_LIST)]
TRAIN_MASK_PATHS = [os.path.join(args.data_path, f"AOI_{i+2}_{city}_Train_processed", "masks") for i, city in enumerate(CITY_LIST)]

TEST_IMG_PATHS = [os.path.join(args.data_path, f"AOI_{i+2}_{city}_Test_public_processed", "RGB-PanSharpen") for i, city in enumerate(CITY_LIST)]
TEST_MASK_PATHS = [os.path.join(args.data_path, f"AOI_{i+2}_{city}_Test_public_processed", "masks") for i, city in enumerate(CITY_LIST)]

CKPT_PATH = f"ckpt/{args.model}.pth"
BEST_CKPT_PATH = f"ckpt/{args.model}_best.pth"

SAMPLE_PATH = f"samples/{args.model}"
RESULT_PATH = f"results/{args.model}"

if os.path.exists(SAMPLE_PATH):
    os.system(f"rm -r {SAMPLE_PATH}")

if os.path.exists(RESULT_PATH):
    os.system(f"rm -r {RESULT_PATH}")
    
os.mkdir(SAMPLE_PATH)
os.mkdir(RESULT_PATH)

IMG_SIZE = (args.size, args.size)
CURRENT_EPOCH = 0

COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 0, 0),
}

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def main():
    
    img_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE)
    ])
    
    dataset = SpaceNet_Dataset(
        img_dir_list=TRAIN_IMG_PATHS, 
        mask_dir_list=TRAIN_MASK_PATHS, 
        img_transform=img_transform,
        mask_transform=mask_transform
    )
    
    train_size = int(len(dataset) * args.train_perc)
    val_size = int(len(dataset) * args.val_perc)
    test_size = len(dataset) - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size]) 
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=2,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True
    )
    

    val_loader = DataLoader(val_dataset,
                              batch_size=args.batch_size,
                              num_workers=2,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True
    )
    
    test_loader = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              num_workers=2,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True
    )
    
    perc = get_label_dist(train_loader)

    model = get_model(args.model, pretrained=True).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    save_checkpoint(model, optimizer, epoch=CURRENT_EPOCH, path=CKPT_PATH)
    
    max_val_loss = 1e10
    while True: 
        ckpt = torch.load(CKPT_PATH)
        epoch = ckpt["epoch"]
        
        train_loss_list = ckpt["train_loss_list"]
        val_loss_list = ckpt["val_loss_list"]

        if epoch == args.epochs:
            break
        
        model = get_model(args.model, pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.to(DEVICE)
        
        opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        opt.load_state_dict(ckpt["opt"])
        
        model.train()
        train_loss = train_fn(train_loader, model, opt, loss_fn, DEVICE)
        train_loss_list.append(train_loss)
        
        model.eval()
        val_loss = val_fn(val_loader, model, loss_fn, DEVICE, COLOR_MAP, SAMPLE_PATH, epoch, perc, viz = epoch % 5 == 0)
        val_loss_list.append(val_loss)
        
        if val_loss < max_val_loss:
            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": epoch,
                },
            BEST_CKPT_PATH)
        
        save_checkpoint(model, opt, epoch + 1, CKPT_PATH, train_loss_list, val_loss_list)
        
        print(epoch, train_loss, val_loss)
    
    best_ckpt = torch.load(BEST_CKPT_PATH)
    
    test_model = get_model(args.model, pretrained=False)
    test_model.load_state_dict(best_ckpt["model"])
    test_model.to(DEVICE)
    test_model.eval()
    
    test_loss, test_iou = test_fn(test_loader, test_model, loss_fn, DEVICE, perc)
    
    print(test_loss, test_iou)
    
    train_losses = np.array(train_loss_list)
    val_losses = np.array(val_loss_list)
    
    np.save(f"{RESULT_PATH}/train_losses.npy", train_losses)
    np.save(f"{RESULT_PATH}/val_losses.npy", val_losses)

if __name__ == "__main__":
    main()
