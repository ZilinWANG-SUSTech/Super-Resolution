import cv2
import random
import torch
import numpy as np

class PairedTransform:
    def __init__(self, phase="train", lq_patch_size=32, scale=4, image_range="0~1"):
        """
        Args:
            phase (str): 'train' or 'val'.
            lq_patch_size (int): Spatial size of the cropped LR patch.
            scale (int): Upsampling scale factor.
        """
        self.phase = phase
        self.lq_patch_size = lq_patch_size
        self.scale = scale
        # Calculate HR patch size based on LR patch size and scale
        self.gt_patch_size = lq_patch_size * scale
        self.image_range = image_range

    def __call__(self, img, gt):
        """
        Applies paired transforms to LR and HR images.
        Args:
            img (np.ndarray): LR image, shape (H, W, C), BGR format.
            gt (np.ndarray): HR image, shape (H, W, C), BGR format.
        Returns:
            dict: Transformed LR and HR tensors.
        """
        # Convert BGR (cv2 default) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

        if self.phase == "train":
            h_lq, w_lq, _ = img.shape
            
            # Paired Random Crop based on LR size
            x0 = random.randint(0, max(0, w_lq - self.lq_patch_size))
            y0 = random.randint(0, max(0, h_lq - self.lq_patch_size))
            
            img = img[y0:y0 + self.lq_patch_size, x0:x0 + self.lq_patch_size, :]
            
            # Map LR coordinates to HR coordinates
            gt_x0, gt_y0 = x0 * self.scale, y0 * self.scale
            gt = gt[gt_y0:gt_y0 + self.gt_patch_size, gt_x0:gt_x0 + self.gt_patch_size, :]

            # Random Flips
            if random.random() < 0.5:
                img = cv2.flip(img, 1)  # Horizontal flip
                gt = cv2.flip(gt, 1)
            if random.random() < 0.5:
                img = cv2.flip(img, 0)  # Vertical flip
                gt = cv2.flip(gt, 0)

            # Random Transpose HW
            if random.random() < 0.5:
                img = cv2.transpose(img)
                gt = cv2.transpose(gt)

        # Convert to Tensor (HWC -> CHW) and normalize to [-1.0, 1.0] for Diffusion models
        if self.image_range == "0~1":
            img_tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            gt_tensor = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).float() / 255.0
        elif self.image_range == "-1~1":
            img_tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 127.5 - 1.0
            gt_tensor = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).float() / 127.5 - 1.0
        else:
            raise ValueError

        return {'img': img_tensor, 'gt': gt_tensor}