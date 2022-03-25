import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import SpaceNet_Dataset
from model import UNET

BASE_DATA_PATH = "/scratch/ssd002/datasets/cv_project/spacenet"
EXAMPLE_DATA_PATH = os.path.join(BASE_DATA_PATH, "AOI_4_Shanghai_Train_processed")
IMG_PATH = os.path.join(BASE_DATA_PATH, "AOI_4_Shanghai_Train_processed", "RGB-PanSharpen")
MASK_PATH = os.path.join(BASE_DATA_PATH, "AOI_4_Shanghai_Train_processed", "masks")

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
IMG_DIM = (256, 256)
DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)
    
    for batch_id, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
            
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix(loss=loss.item())   

def main():
    normalize = transforms.Normalize(mean=[0], std=[1])
        
    data_transform = transforms.Compose([
        transforms.Resize(IMG_DIM),
            transforms.ToTensor(),
            normalize
    ])

    train_ds = SpaceNet_Dataset(img_dir=IMG_PATH, 
                                mask_dir=MASK_PATH, 
                                transform=data_transform)
    
    train_loader = DataLoader(train_ds,
                              batch_size=BATCH_SIZE,
                              num_workers=2,
                              pin_memory=True,
                              shuffle=True
                              )
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == '__main__':
    main()