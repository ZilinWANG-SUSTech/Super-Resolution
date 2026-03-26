import os
import cv2
from torch.utils.data import Dataset

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