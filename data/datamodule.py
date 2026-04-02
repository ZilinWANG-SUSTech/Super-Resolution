import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .dataset import BasicSRDataset
from .transforms import PairedTransform
import os

class SRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_img_dir: str,
        train_gt_dir: str,
        val_img_dir: str,
        val_gt_dir: str,
        test_img_dir: str = None,
        test_gt_dir: str = None,
        batch_size: int = 8,
        num_workers: int = 4,
        lq_patch_size: int = 32,
        scale: int = 4,
        image_range: str = "0~1",
    ):
        """
        Args:
            train_img_dir (str): Path to training LR images.
            train_gt_dir (str): Path to training HR images.
            val_img_dir (str): Path to validation LR images.
            val_gt_dir (str): Path to validation HR images.
            test_img_dir (str, optional): Path to testing LR images.
            test_gt_dir (str, optional): Path to testing HR images.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            lq_patch_size (int): Crop size for LR images.
            scale (int): Upsampling scale factor.
        """
        super().__init__()
        self.train_img_dir = os.path.join(train_img_dir, f"X{scale}")
        self.train_gt_dir = train_gt_dir
        self.val_img_dir = os.path.join(val_img_dir, f"X{scale}")
        self.val_gt_dir = val_gt_dir
        self.test_img_dir = os.path.join(test_img_dir, f"X{scale}")
        self.test_gt_dir = test_gt_dir
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lq_patch_size = lq_patch_size
        self.scale = scale
        self.image_range = image_range

    def setup(self, stage: str = None):
        """
        Instantiates datasets and transforms based on the execution stage.
        """
        if stage == 'fit' or stage is None:
            train_transform = PairedTransform(
                phase="train", lq_patch_size=self.lq_patch_size, scale=self.scale, image_range=self.image_range,
            )
            val_transform = PairedTransform(
                phase="val", lq_patch_size=self.lq_patch_size, scale=self.scale, image_range=self.image_range,
            )
            
            self.train_dataset = BasicSRDataset(
                img_dir=self.train_img_dir, 
                gt_dir=self.train_gt_dir, 
                transform=train_transform
            )
            
            self.val_dataset = BasicSRDataset(
                img_dir=self.val_img_dir, 
                gt_dir=self.val_gt_dir, 
                transform=val_transform
            )
            
        if stage == 'test' or stage is None:
            if self.test_img_dir is not None and self.test_gt_dir is not None:
                # Use "val" phase transform for testing (no random crops/flips)
                test_transform = PairedTransform(
                    phase="val", lq_patch_size=self.lq_patch_size, scale=self.scale, image_range=self.image_range,
                )
                self.test_dataset = BasicSRDataset(
                    img_dir=self.test_img_dir, 
                    gt_dir=self.test_gt_dir, 
                    transform=test_transform
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        if hasattr(self, 'test_dataset'):
            return DataLoader(
                self.test_dataset, 
                batch_size=1,  
                shuffle=False, 
                num_workers=self.num_workers,
                pin_memory=True
            )
        raise ValueError("Test datasets are not configured.")