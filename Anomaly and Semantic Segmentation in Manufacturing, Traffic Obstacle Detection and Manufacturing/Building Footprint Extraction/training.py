import torch, numpy as np
from utils import save_viz
from metrics import iou

def get_label_dist(loader):
    count_list = []
    for _, (_, lbl) in enumerate(loader):
        cnt = torch.bincount(lbl.int().flatten())
        count_list.append(cnt)

    cnts = torch.stack(count_list, dim=0).sum(dim=0).tolist()
    zero_count, one_count = cnts[0], cnts[1]
    perc = zero_count / (zero_count + one_count)
    return perc


def train_fn(loader, model, opt, loss_fn, device):
    loss_list = []
    for batch_id, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.float().to(device)
        predictions = model(data)['out']
        loss = loss_fn(predictions, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_list.append(loss.item())

    mean_loss = np.mean(loss_list)
    return mean_loss


def val_fn(loader, model, loss_fn, device, color_map, sample_path, epoch, perc, viz):
    loss_list = []
    for batch_id, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.float().to(device=device)
        with torch.no_grad():
            predictions = model(data)['out']
            loss = loss_fn(predictions, targets)
        loss_list.append(loss.item())
        if viz:
            save_viz(data, predictions, targets, color_map, epoch, sample_path, perc)
            viz = False

    mean_loss = np.mean(loss_list)
    return mean_loss


def test_fn(loader, model, loss_fn, device, perc):
    target_list, pred_list = [], []
    for batch_id, (data, targets) in enumerate(loader):
        data = data.to(device=device)
        targets = targets.float().to(device=device)
        with torch.no_grad():
            pred = model(data)['out']
        pred_list.append(pred)
        target_list.append(targets)

    pred = torch.cat(pred_list, dim=0)
    target = torch.cat(target_list, dim=0)
    thresh = np.quantile(pred.flatten().cpu().numpy(), perc)
    test_loss = loss_fn(pred, target).item()
    pred = (pred > thresh).float()
    test_iou = iou(pred, target).item()
    return (
     test_loss, test_iou)