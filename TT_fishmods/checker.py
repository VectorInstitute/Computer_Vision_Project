import torch
MVTEC_ROOT_DIR = "/scratch/ssd002/datasets/MVTec_AD"
from vector_cv_tools import transforms as VT
import albumentations as A
from albumentations.pytorch import ToTensorV2
from vector_cv_tools import datasets as vdatasets

#from tqdm.notebook import tqdm
from tqdm import tqdm
from random import seed, sample

basic_transform = VT.ComposeMVTecTransform([A.Resize(128, 128), A.ToFloat(max_value=255), ToTensorV2()])

from vector_cv_tools.datasets.mvtec import MVTec_OBJECTS

import torch 
from sklearn.metrics import roc_auc_score#Get Error Maps 

#Oh! That's right, I'm discarding the random deviation while vaeing. That might not be the normal thing to do...
# Anyway, these functions are helpers that encode and decode.
def imgs2vecs(model,img):
    with torch.no_grad():
        z = model.module.encoder(img)
    return z[..., :100] # we only like mu
def vecs2imgs(model,vec):
    with torch.no_grad():
        img = model.module.decoder(vec.unsqueeze(-1).unsqueeze(-1))
    return img
def imgs2imgs(model,img):
    return vecs2imgs(
                model,
                imgs2vecs(
                    model,
                    img
                )
    )

#TODO: this would probs be more efficient if the flat_segmentation_predictions were calc'd ahead of time.
def get_auc(sample_tensor, reconstructions, masks_tensor):
    error_maps = torch.abs(reconstructions - sample_tensor)

    #Get Error Maps with 1 channel
    error_maps = torch.sum(error_maps, dim=1)   #Flatten Error Maps to vector

    flat_error_maps = torch.flatten(error_maps)#Flatten Ground Truth Segmentation labels to vector 
    flat_labels = torch.flatten(masks_tensor)
    #Scale Error Maps between 0 and 1 to yield segmentation prediction
    flat_segmentation_predictions = (
        ( flat_error_maps - torch.min(flat_error_maps) ) / 
        (torch.max(flat_error_maps) - torch.min(flat_error_maps) )
    )
    res = roc_auc_score(flat_labels.cpu().numpy(), flat_segmentation_predictions.cpu().numpy())
    
    return(res)


class Checker():
    def __init__(self):

        dsets = list(vdatasets.MVTec(MVTEC_ROOT_DIR, split="test", transforms = basic_transform,
                                     obj_types=[obj_type]) for obj_type in MVTec_OBJECTS)

        good_mask = torch.zeros_like(dsets[0][0][1]['mask'])


        goods_w_masks=[]
        bads_w_masks=[]
        for dset in tqdm(dsets):
            goods_w_masks+=[[]]
            bads_w_masks+=[[]]
            for thingamajig in dset:
                if "good" == thingamajig[1]['label_names'][1]:
                    goods_w_masks[-1]+=[[thingamajig[0].unsqueeze(0),
                                        good_mask.unsqueeze(0).unsqueeze(0)]]
                else:
                    bads_w_masks[-1]+=[[thingamajig[0].unsqueeze(0),
                                       thingamajig[1]['mask'].unsqueeze(0).unsqueeze(0)]]


        sample_size_per_ob = 10
        seed(42) # the same random sample every time. Change if you want a different random sampling.
        img_samples = []
        mask_samples = []
        for obs in goods_w_masks:
            samples = sample(obs,sample_size_per_ob)
            for img, mask in samples:
                img_samples += [img]
                mask_samples += [mask]
        for obs in bads_w_masks:
            samples = sample(obs,sample_size_per_ob)
            for img, mask in samples:
                img_samples += [img]
                mask_samples += [mask]

        sample_tensor = torch.cat(img_samples,0).cuda()
        masks_tensor = torch.cat(mask_samples,0).cuda()

        self.X = sample_tensor
        self.masks = masks_tensor


    def xloss_zloss_auc(self, model):
        model.eval()

        X = self.X
        Z  = imgs2vecs(model, X)
        Xr = vecs2imgs(model, Z)
        Zr = imgs2vecs(model, Xr)

        B = X.size(0)
        MSEx = (X - Xr).pow(2).sum() / B
        MSEz = (Z - Zr).pow(2).sum() / B
        auc = get_auc(self.X, Xr, self.masks)

        MSEx = float(MSEx)
        MSEz = float(MSEz)
  
        # we can safely assume this will be called from within
        # a training session, so it's probably kind to return
        # the model to training mode.
        model.train()

        return( MSEx, MSEz, auc )



