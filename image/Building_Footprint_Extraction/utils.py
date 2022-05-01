import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead

import numpy as np
import matplotlib.pyplot as plt

from model import UNET, UNETPlus

def get_model(model_type, pretrained):
    model = None
    if model_type == "fcn50":
        model = get_model_fcn50(pretrained)
        
    elif model_type == "fcn101":
        model = get_model_fcn101(pretrained)
     
    elif model_type == "dlv350": 
        model = get_model_dlv350(pretrained)
    
    elif model_type == "dlv3101": 
        model = get_model_dlv3101(pretrained)
     
    elif model_type == "unet":
        model = UNET(in_channels=3, out_channels=1)
     
    elif model_type == "unetplus":
        model = UNETPlus(n_channels=3, n_classes=1)
    
    return model

def get_model_fcn50(pretrained=True, c_out=1):
    # Prepare Model and Save to Checkpoint Directory
    model = fcn_resnet50(pretrained=pretrained)

    model.classifier = FCNHead(2048, c_out)
    model.aux_classifier = None
    model = nn.DataParallel(model)
    return model

def get_model_fcn101(pretrained=True, c_out=1):
    # Prepare Model and Save to Checkpoint Directory
    model = fcn_resnet101(pretrained=pretrained)
    model.classifier = FCNHead(2048, c_out)
    model.aux_classifier = None
    model = nn.DataParallel(model)
    return model

def get_model_dlv350(pretrained=True, c_out=1):
    # Prepare Model and Save to Checkpoint Directory
    model = deeplabv3_resnet50(pretrained=pretrained)
    model.classifier = DeepLabHead(2048, c_out)
    model.aux_classifier = None
    model = nn.DataParallel(model)

    return model

def get_model_dlv3101(pretrained=True, c_out=1):
    # Prepare Model and Save to Checkpoint Directory
    model = deeplabv3_resnet101(pretrained=pretrained)
    model.classifier = DeepLabHead(2048, c_out)
    model.aux_classifier = None
    model = nn.DataParallel(model)
    return model

def save_checkpoint(model, opt, epoch, path, train_loss_list=[], val_loss_list=[]):
    """Save Checkpoint"""

    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
        "train_loss_list": train_loss_list,
        "val_loss_list": val_loss_list
        },
        path)


def save_viz(img, out, lbl, color_map, epoch, sample_path, perc):
    img = img.cpu().numpy()
    out = out.cpu().numpy()
    lbl = lbl.cpu().numpy()
    
    thresh = np.quantile(out, perc)
    
    print("thresh", thresh) 
    
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    rows = out.shape[2]
    cols = out.shape[3]

    masks = []
    masks_gt = []
    for index, (im, o, l) in enumerate(zip(img, out, lbl)):
        o, l = o.squeeze(), l.squeeze()
        
        o = (o > thresh).astype(int) 
 

        mask = np.zeros((rows, cols, 3), dtype=np.uint8)
        mask_gt = np.zeros((rows, cols, 3), dtype=np.uint8)

        for j in range(rows):
            for i in range(cols):
                mask[j, i] = color_map[o[j, i]]
                mask_gt[j, i] = color_map[l[j, i]]
        


        f, axarr = plt.subplots(1, 3, figsize=(20, 20))
        im = np.moveaxis(im, 0, -1)
        axarr[0].imshow(im)
        axarr[0].title.set_text('Image')
        axarr[1].imshow(mask_gt)
        axarr[1].title.set_text('Label')
        axarr[2].imshow(mask)
        axarr[2].title.set_text('Prediction')
        f.savefig( f"{sample_path}/epoch_{str(epoch)}_{str(index)}.jpg")