import os
import torch
import glob
from PIL import Image
import numpy as np

from torch.utils.data import Dataset

class MVTecADDataset(Dataset):
    def __init__(self, img_dir, mode, transform, size=128):
        self.img_dir = img_dir
        self.mode = mode
        self.size = size

        if self.mode == "train":
            self.img_paths = glob.glob(f"{self.img_dir}/train/good/*.png")
        else:

            paths = glob.glob(f"{self.img_dir}/test/*/*.png")
            inlier_img_paths = glob.glob(f"{self.img_dir}/test/good/*.png")
            outlier_img_paths = list(set(paths) - set(inlier_img_paths))
            self.img_paths = inlier_img_paths + outlier_img_paths
            self.outlier_lbl_paths = [f"{self.img_dir}/ground_truth/{path.split('/')[-2]}/{path.split('/')[-1][:-4]}_mask.png" for path in outlier_img_paths]
            self.outlier_lbl = np.array([np.array(Image.open(path).convert('1').resize((self.size, self.size))) for path in self.outlier_lbl_paths])
        
            self.inlier_lbl = np.zeros(shape=(len(inlier_img_paths), self.outlier_lbl.shape[1], self.outlier_lbl.shape[2]))
            
            self.labels = torch.from_numpy(np.concatenate([self.inlier_lbl, self.outlier_lbl])).int()
  

        self.transform = transform

    def __getitem__(self, index):
        if self.mode == "test":
            x = Image.open(self.img_paths[index]).convert("RGB")
            if self.transform is not None:
                x = self.transform(x)
            

            y = self.labels[index]
            return x, y
        else:
            x = Image.open(self.img_paths[index]).convert('RGB')
            if self.transform is not None:
                x = self.transform(x)
            return x

    def __len__(self):
        return len(self.img_paths)