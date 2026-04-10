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
    

class GuidedTransform:
    def __init__(self, phase="train", lq_patch_size=32, scale=4, image_range="-1~1"):
        """
        Args:
            phase (str): 'train' or 'val'.
            lq_patch_size (int): Spatial size of the cropped LR patch.
            scale (int): Upsampling scale factor.
            image_range (str): Normalization range ('0~1' or '-1~1').
        """
        self.phase = phase
        self.lq_patch_size = lq_patch_size
        self.scale = scale
        # Calculate HR and Guide patch size based on LR patch size and scale
        self.gt_patch_size = lq_patch_size * scale
        self.image_range = image_range

    def __call__(self, lr, hr, guide):
        """
        Applies paired transforms to LR, HR, and Guide images symmetrically.
        Args:
            lr (np.ndarray): LR image, shape (H_lr, W_lr, C), BGR format.
            hr (np.ndarray): HR image, shape (H_hr, W_hr, C), BGR format.
            guide (np.ndarray): Guide image, shape (H_hr, W_hr, C), BGR format.
        Returns:
            dict: Transformed LR, HR, and Guide tensors.
        """
        # Convert BGR (cv2 default) to RGB for all modalities
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        guide = cv2.cvtColor(guide, cv2.COLOR_BGR2RGB)

        if self.phase == "train":
            h_lq, w_lq, _ = lr.shape
            
            # Paired Random Crop based on LR size
            x0 = random.randint(0, max(0, w_lq - self.lq_patch_size))
            y0 = random.randint(0, max(0, h_lq - self.lq_patch_size))
            
            lr = lr[y0:y0 + self.lq_patch_size, x0:x0 + self.lq_patch_size, :]
            
            # Map LR coordinates to HR and Guide coordinates
            gt_x0, gt_y0 = x0 * self.scale, y0 * self.scale
            
            # Apply exact same high-res crop to both HR and Guide
            hr = hr[gt_y0:gt_y0 + self.gt_patch_size, gt_x0:gt_x0 + self.gt_patch_size, :]
            guide = guide[gt_y0:gt_y0 + self.gt_patch_size, gt_x0:gt_x0 + self.gt_patch_size, :]

            # Random Flips (Symmetric application)
            if random.random() < 0.5:
                lr = cv2.flip(lr, 1)  # Horizontal flip
                hr = cv2.flip(hr, 1)
                guide = cv2.flip(guide, 1)
            if random.random() < 0.5:
                lr = cv2.flip(lr, 0)  # Vertical flip
                hr = cv2.flip(hr, 0)
                guide = cv2.flip(guide, 0)

            # Random Transpose HW (Symmetric application)
            if random.random() < 0.5:
                lr = cv2.transpose(lr)
                hr = cv2.transpose(hr)
                guide = cv2.transpose(guide)

        # Convert to Tensor (HWC -> CHW) and normalize
        if self.image_range == "0~1":
            lr_tensor = torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1).float() / 255.0
            hr_tensor = torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1).float() / 255.0
            guide_tensor = torch.from_numpy(np.ascontiguousarray(guide)).permute(2, 0, 1).float() / 255.0
        elif self.image_range == "-1~1":
            # Typically used for Diffusion Models
            lr_tensor = torch.from_numpy(np.ascontiguousarray(lr)).permute(2, 0, 1).float() / 127.5 - 1.0
            hr_tensor = torch.from_numpy(np.ascontiguousarray(hr)).permute(2, 0, 1).float() / 127.5 - 1.0
            guide_tensor = torch.from_numpy(np.ascontiguousarray(guide)).permute(2, 0, 1).float() / 127.5 - 1.0
        else:
            raise ValueError(f"Unsupported image_range configuration: {self.image_range}")

        return {'lr': lr_tensor, 'hr': hr_tensor, 'guide': guide_tensor}