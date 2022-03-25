import os
from PIL import Image
from torch.utils.data.dataset import Dataset

class SpaceNet_Dataset(Dataset): 
    def __init__(self, img_dir, mask_dir, transform = None):
        self.img_dir = img_dir 
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)
        self.masks = os.listdir(mask_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask