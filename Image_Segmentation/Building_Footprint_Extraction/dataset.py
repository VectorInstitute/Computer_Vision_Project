

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

class SpaceNet_Dataset(Dataset): 
    def __init__(self, img_dir_list, mask_dir_list, img_transform = None, mask_transform=None):
        self.img_dir_list = img_dir_list
        self.mask_dir_list = mask_dir_list
        
        img_paths, mask_paths = [], []
        
        for img_dir, mask_dir in zip(img_dir_list, mask_dir_list):
            img_paths += [f"{img_dir}/{img_file}" for img_file in os.listdir(img_dir)]
            mask_paths += [f"{mask_dir}/{mask_file}" for mask_file in os.listdir(mask_dir)]
        
        self.img_paths = sorted(img_paths) 
        self.mask_paths = [f"{new_mask_path}_mask.png" for new_mask_path in sorted([mask_path[:-9] for mask_path in mask_paths])]
        
        self.img_transform = img_transform
        self.mask_transform = mask_transform

        
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        
        img = Image.open(img_path)
        img = self.img_transform(img) 
        
        mask = Image.open(mask_path).convert("1")
        mask = self.mask_transform(mask)
        mask = torch.from_numpy(np.array(mask).astype(int)).unsqueeze(0)
       
        return img, mask