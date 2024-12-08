import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# define class to create dataset
class CarDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and mask file name
        img_name = self.images[idx]
        mask_name = f"{os.path.splitext(img_name)[0]}_mask.gif"
        
        # Get full file paths
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Apply transformations (if any)
        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform: 
            mask = self.mask_transform(mask)

        # Convert mask to binary tensor (0 and 1)
        mask = np.array(mask, dtype=np.float32)
        mask = (mask > 0).astype(np.float32) # Normalize to 0 and 1 (otherwise they are 0/125 and 1/125)
        mask = torch.tensor(mask) 

        return image, mask

