# system imports
import os
import logging
import glob
from pathlib import Path
import re
import argparse

# external dependencies
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import DataLoader

# relative imports
from model import AE, ConvVAE, ae_loss_fn, vae_loss_fn
from datasets import MVTecADDataset
from utils import train_step, test_step, save_checkpoint

parser = argparse.ArgumentParser(description="Feature Memory for Anomaly Detection")

# basic config
parser.add_argument('--model', type=str, help='Architecture variation for experiments. ae or vae.')
parser.add_argument('--epochs', type=int,  default=100, help=' The number of epochs to train the model.')
parser.add_argument('--batch_size', type=int,  default=8, help=' The batch size for training, validation and testing.')
parser.add_argument('--learning_rate', type=float,  default=.001, help='Learning rates of model.')
parser.add_argument('--size', type=int,  default=128, help='Side length of input image')
parser.add_argument('--data_path', type=str, help='The root directory of the dataset.')
parser.add_argument('--ckpt_path', type=str, help='The directory to save model checkpoints.')

args = parser.parse_args()

# Data Paths

CLASSES =  ["toothbrush",
            "pill",
           "leather", 
           "hazelnut", 
           "capsule", 
           "cable", 
           "bottle", 
           "zipper", 
           "tile", 
           "transistor", 
           "wood", 
           "metal_nut", 
           "screw", 
           "carpet",  
           "grid"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():

    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Resize(size=(args.size, args.size)),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    test_auc_list = []
    for inlier in CLASSES: 
        # Prepare Data
        print("class", inlier)
        current_epoch = 0
        ckpt_path = f"{args.ckpt_path}/{inlier}.pth"
        img_dir = f"{args.data_path}/{inlier}"
        train_dataset = MVTecADDataset(img_dir, "train", transform)
        test_dataset = MVTecADDataset(img_dir, "test", transform, args.size)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
        
        
        model = ConvVAE() if args.model == "vae" else AE()
        model = torch.nn.DataParallel(model)
        
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        save_checkpoint(model, optimizer, epoch=current_epoch, path=ckpt_path)
        
        loss_fn = vae_loss_fn if args.model == "vae" else ae_loss_fn
        
        highest_auc = 0
        while True:
            ckpt = torch.load(ckpt_path)
            epoch = ckpt["epoch"]
            
            if epoch == args.epochs:
                break
            
            model = ConvVAE() if args.model == "vae" else AE()
            model = nn.DataParallel(model)
            model.load_state_dict(ckpt["model"])
            model.to(DEVICE)
     
            model.train()
            train_loss = train_step(train_loader, model, optimizer, loss_fn, DEVICE, args.model)
            
            model.eval()
            test_auc, test_loss = test_step(test_loader, model, loss_fn, DEVICE, args.model)
            
            print(f"Train Loss: {str(train_loss)} \t Test AUC: {str(test_auc)}")
            
            if test_auc > highest_auc: 
                highest_auc = test_auc
            
            save_checkpoint(model, optimizer, epoch + 1, ckpt_path)
                  
        test_auc_list.append(highest_auc)


    print(f"Average AUC: {str(np.mean(test_auc_list))}")

######################################################

if __name__ == "__main__":
    main()