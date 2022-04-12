import tqdm 

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision.models.segmentation import fcn_resnet50


def train_step(model, opt, criterion, dataloader, epoch, device):
    losses = []
    counter = 0
    for i, (img, lbl) in enumerate(dataloader):
        lbl = lbl.long()
        img, lbl = img.to(device), lbl.to(device)
        opt.zero_grad()
        out = model(img)["out"]
        loss = criterion(out, lbl)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        
    return np.mean(losses)   

def val_step(model, criterion, dataloader, epoch, device, lf_map, sample_path):
    losses = []
    dices = []
    viz = True
    for i, (img, lbl) in enumerate(dataloader):
        lbl = lbl.long()
        img, lbl = img.to(device), lbl.to(device)
        
        with torch.no_grad():
            out = model(img)["out"]

        loss = criterion(out, lbl)
        losses.append(loss.item())

        if viz:
            save_viz(img, out, lbl, lf_map, epoch, sample_path) 
            viz = False 
         
    return np.mean(losses)

def save_viz(img, out, lbl, color_map, epoch, sample_path):
    img = img.cpu().numpy()
    out = out.cpu().numpy()
    lbl = lbl.cpu().numpy()
    rows = out.shape[2]
    cols = out.shape[3]
    
    masks = []
    masks_gt = []
    for index, (im, o, l) in enumerate(zip(img, out, lbl)):
        mask = np.zeros((rows, cols, 3), dtype=np.uint8)
        mask_gt = np.zeros((rows, cols, 3), dtype=np.uint8)
        for j in range(rows):
            for i in range(cols):
                mask[j, i] = color_map[np.argmax(o[:, j, i]-1, axis=0)]
                mask_gt[j, i] = color_map[l[j, i]]
                
        mask_path = f"{sample_path}/epoch_{str(epoch)}_pred_{str(index)}.jpg"
        lbl_path = f"{sample_path}/epoch_{str(epoch)}_lbl_{str(index)}.jpg"
        img_path = f"{sample_path}/epoch_{str(epoch)}_img_{str(index)}.jpg"
        f, axarr = plt.subplots(1, 3, figsize=(20, 20))
        im = np.moveaxis(im, 0, -1)
        axarr[0].imshow(im)
        axarr[0].title.set_text('Image')
        axarr[1].imshow(mask_gt)
        axarr[1].title.set_text('Label')
        axarr[2].imshow(mask)
        axarr[2].title.set_text('Prediction')
        f.savefig( f"{sample_path}/epoch_{str(epoch)}_{str(index)}.jpg")

def get_model(pretrained=False):
    # Prepare Model and Save to Checkpoint Directory
    model = fcn_resnet50(pretrained=pretrained)
    model.classifier[4] = nn.Conv2d(512, 4, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1))
    model.aux_classifier = None
    model = nn.DataParallel(model)
    return model

def save_checkpoint(model, opt, epoch, path):
    """Save Checkpoint"""

    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
        },
        path)
