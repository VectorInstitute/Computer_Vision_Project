import torch
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve

def get_auc(preds, lbls):
    preds = preds.flatten().cpu().numpy()
    lbls = lbls.flatten().cpu().numpy()
    
    auc = roc_auc_score(lbls, preds)
    return auc

def save_checkpoint(model, opt, epoch, path):
    """Save Checkpoint"""

    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch
        },
        path)
    

def train_step(loader, model, optimizer, loss_fn, device, model_str):
    
    train_loss_list = []
    
    for i, data in enumerate(loader): 
        data = data.to(device) 
        optimizer.zero_grad()
        if model_str == "vae":
            recon, mu, logvar = model(data)
            loss = loss_fn(data, recon, mu, logvar)
        else: 
            recon = model(data) 
            loss = loss_fn(data, recon)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
        
    return np.mean(train_loss_list)

def test_step(loader, model, loss_fn, device, model_str):
    
    loss_list, error_map_list, lbl_list = [], [], []
    for i, (data, lbl) in enumerate(loader): 
        data, lbl = data.to(device), lbl.to(device)
        
        with torch.no_grad():
            if model_str == "vae":
                recon, mu, logvar = model(data)
                loss = loss_fn(data, recon, mu, logvar)
            else: 
                recon = model(data) 
                loss = loss_fn(data, recon)
        loss_list.append(loss.item())
        error_map = torch.mean((data - recon)**2, dim=1).unsqueeze(1) 
        error_map_list.append(error_map)
        lbl_list.append(lbl)

    error_maps = torch.cat(error_map_list, dim=0)
    lbls = torch.cat(lbl_list, dim=0)
    preds = (error_maps - torch.min(error_maps)) / (torch.max(error_maps) - torch.min(error_maps))
    
    auc = get_auc(preds, lbls)
    loss = np.mean(loss_list)
    
    return auc, loss
        