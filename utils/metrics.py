import os
import tempfile
import torch
import torch.nn as nn
from torchvision.utils import save_image
import pyiqa
import pandas as pd


class SREvaluatorPyIQA(nn.Module):
    def __init__(self, crop_border: int = 4, test_y_channel: bool = True, device: str = 'cuda'):
        """
        Args:
            crop_border (int): Number of pixels to crop from the edge before evaluation 
                               (usually equals the scale factor).
            test_y_channel (bool): Whether to calculate PSNR and SSIM on the Y channel in YCbCr space.
            device (str): Device to run the evaluation on ('cuda' or 'cpu').
        """
        super().__init__()
        self.crop_border = crop_border
        self.device = torch.device(device)
        self.test_y_channel = test_y_channel
        
        # 1. Initialize Full-Reference (FR) metrics
        # pyiqa perfectly aligns with MATLAB's color space conversion internally.
        # We can directly pass RGB tensors without manual BGR conversion.
        self.psnr = pyiqa.create_metric('psnr', test_y_channel=test_y_channel).to(self.device)
        self.ssim = pyiqa.create_metric('ssim', test_y_channel=test_y_channel).to(self.device)
        self.lpips = pyiqa.create_metric('lpips').to(self.device)
        self.dists = pyiqa.create_metric('dists').to(self.device)
        
        # 2. Initialize No-Reference (NR) metrics
        self.niqe = pyiqa.create_metric('niqe').to(self.device)
        
        # 3. Initialize distribution-based metric (FID)
        self.fid = pyiqa.create_metric('fid').to(self.device)
        
        # Temporary directories for FID evaluation.
        # FID must be calculated over the entire dataset distribution, not averaged across batches.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_preds_dir = os.path.join(self.temp_dir.name, 'preds')
        self.temp_target_dir = os.path.join(self.temp_dir.name, 'target')
        os.makedirs(self.temp_preds_dir, exist_ok=True)
        os.makedirs(self.temp_target_dir, exist_ok=True)

        self.reset()

    def reset(self):
        """Clears accumulators at the beginning/end of an epoch."""
        self.metrics_sum = {
            'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0, 'dists': 0.0, 'niqe': 0.0
        }
        self.total_samples = 0
        self.img_idx = 0 # Used for naming cached images for FID
        
        # Clear historical images in the FID temporary directories
        for folder in [self.temp_preds_dir, self.temp_target_dir]:
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

    @torch.no_grad()
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Processes a single batch and accumulates metrics.
        Args:
            preds: Tensor of shape (B, C, H, W) in range [0.0, 1.0], RGB order.
            target: Tensor of shape (B, C, H, W) in range [0.0, 1.0], RGB order.
        """
        # Ensure data is on the correct device and clamped to the valid range
        preds = torch.clamp(preds, 0.0, 1.0).to(self.device)
        target = torch.clamp(target, 0.0, 1.0).to(self.device)
        batch_size = preds.size(0)

        # 1. Cache images for global FID calculation
        for i in range(batch_size):
            save_image(preds[i], os.path.join(self.temp_preds_dir, f"{self.img_idx:05d}.png"))
            save_image(target[i], os.path.join(self.temp_target_dir, f"{self.img_idx:05d}.png"))
            self.img_idx += 1

        # 2. Tensor-level Crop Border processing (efficient slicing, replacing numpy logic)
        if self.crop_border > 0:
            c = self.crop_border
            preds = preds[..., c:-c, c:-c]
            target = target[..., c:-c, c:-c]

        # 3. Calculate batch metrics (pyiqa supports direct batch tensor input)
        # Use .item() to extract scalar values and prevent memory leaks from the computation graph
        batch_metrics = {
            'psnr': self.psnr(preds, target).mean().item(),
            'ssim': self.ssim(preds, target).mean().item(),
            'lpips': self.lpips(preds, target).mean().item(),
            'dists': self.dists(preds, target).mean().item(),
            'niqe': self.niqe(preds).mean().item()  # NIQE is a no-reference metric, requires only preds
        }

        # 4. Global accumulation (multiply by batch_size to ensure mathematically correct averaging later)
        for k, v in batch_metrics.items():
            self.metrics_sum[k] += v * batch_size
            
        self.total_samples += batch_size

        return batch_metrics

    def compute(self) -> dict:
        """Calculates global metrics for all samples at the end of the epoch."""
        if self.total_samples == 0:
            return {k: 0.0 for k in self.metrics_sum.keys()}
        
        # 1. Calculate global average for all scalar metrics
        final_metrics = {
            k: round(v / self.total_samples, 4) for k, v in self.metrics_sum.items()
        }
        
        # 2. Calculate global FID (read all cached images from the epoch for distribution statistics)
        # Note: FID relies on feature distribution and usually requires hundreds of images to be statistically meaningful.
        final_metrics['fid'] = round(self.fid(self.temp_preds_dir, self.temp_target_dir).item(), 4)
        
        return final_metrics
        
    def __del__(self):
        """Automatically cleans up temporary directories upon destruction."""
        self.temp_dir.cleanup()

    def save_to_excel(self, save_path: str, metrics: dict = None):
        """
        Exports the final summary to an Excel file.
        Args:
            metrics (dict, optional): If provided, use these metrics instead of re-computing.
        """
        # If metrics are not provided, compute them now
        summary = metrics if metrics is not None else self.compute()
        
        # Convert dictionary to a single-row DataFrame
        df = pd.DataFrame([summary])
        df.insert(0, 'Stage', 'Test_Global_Average')
        
        df.to_excel(save_path, index=False)
        print(f"Final metrics saved to {save_path}")