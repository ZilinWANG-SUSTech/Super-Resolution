import os
import cv2
from torch.utils.data import Dataset
import torch
import numpy as np


class BasicSRDataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        """
        Args:
            img_dir (str): Path to the directory containing Low-Resolution (LR) images.
            gt_dir (str): Path to the directory containing High-Resolution (HR) ground truth images.
            transform (callable, optional): Transform pipeline to apply.
        """
        super().__init__()
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        
        # Sort to ensure matching pairs
        self.image_names = sorted(os.listdir(self.gt_dir))
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        gt_path = os.path.join(self.gt_dir, img_name)
        
        # Read images using cv2 (BGR format)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)

        if img is None or gt is None:
            raise ValueError(f"Failed to read image pair: {img_name}")

        # Apply paired transforms
        if self.transform is not None:
            processed = self.transform(img, gt)
            img_tensor = processed['img']
            gt_tensor = processed['gt']
        else:
            raise ValueError("Transform pipeline must be provided.")

        return {
            "img": img_tensor,
            "gt": gt_tensor,
            "img_name": img_name
        }
    


class GuidedSRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, guide_dir, transform=None):
        """
        Args:
            lr_dir (str): Path to the directory containing Low-Resolution Pol-SAR images.
            hr_dir (str): Path to the directory containing High-Resolution Pol-SAR ground truth.
            guide_dir (str): Path to the directory containing RGB Guide images.
            transform (callable, optional): Transform pipeline to apply across all modalities.
        """
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.guide_dir = guide_dir
        self.transform = transform
        
        # Sort to ensure matching pairs, assuming filenames are identical across directories
        self.image_names = sorted(os.listdir(self.hr_dir))
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        lr_path = os.path.join(self.lr_dir, img_name)
        hr_path = os.path.join(self.hr_dir, img_name)
        guide_path = os.path.join(self.guide_dir, img_name)
        
        # Read images using cv2 (BGR format) for 3-channel data
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR)
        guide_img = cv2.imread(guide_path, cv2.IMREAD_COLOR)

        if lr_img is None or hr_img is None or guide_img is None:
            raise ValueError(f"Failed to read image trio: {img_name}. Check if it exists in all directories.")

        # Apply paired transforms
        if self.transform is not None:
            # Assuming transform uses keyword arguments like albumentations
            processed = self.transform(lr=lr_img, hr=hr_img, guide=guide_img)
            lr_tensor = processed['lr']
            hr_tensor = processed['hr']
            guide_tensor = processed['guide']
        else:
            raise ValueError("Transform pipeline must be provided.")

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "guide": guide_tensor,
            "img_name": img_name
        }