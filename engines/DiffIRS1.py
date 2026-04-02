import importlib
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.metrics import SREvaluatorPyIQA
from utils import ENGINE_REGISTRY, build_network
import os


@ENGINE_REGISTRY.register()
class DiffIRS1LightningModule(pl.LightningModule):
    def __init__(
        self, 
        network_config: dict,
        optimizer_config: dict,
        eval_crop_border: int = 4, # Exposed for YAML configuration
        lr_scheduler_config: dict = None,
        img_size: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config

        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=True)
        self.net = build_network(network_config)
        

    def forward(self, lr_img: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        hr = batch['gt']
        lr = batch['img']
        # Regression models predict the image directly
        output, _ = self.net(lr, hr)
        # Handle diffusers UNet output wrapper if used, else use tensor directly
        preds = output.sample if hasattr(output, 'sample') else output
        
        # L1 Loss is mathematically strictly preferred over MSE for PSNR/SSIM optimization in standard SR
        loss = F.l1_loss(preds, hr)
        
        with torch.no_grad():
            preds_eval = torch.clamp(preds.detach(), 0.0, 1.0)
            hr_eval = torch.clamp(hr.detach(), 0.0, 1.0)
            train_psnr = self.evaluator.psnr(preds_eval, hr_eval).mean().item()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/psnr_epoch", train_psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        hr = batch['gt']
        lr = batch['img']
        
        # Calculate padding needed to make LR dimensions multiples of 16
        _, _, h_lr, w_lr = lr.size()
        scale = hr.size(2) // h_lr
        pad_mult = 16
        pad_h = (pad_mult - h_lr % pad_mult) % pad_mult
        pad_w = (pad_mult - w_lr % pad_mult) % pad_mult
        # Apply 'reflect' padding if needed
        if pad_h > 0 or pad_w > 0:
            lr_padded = F.pad(lr, (0, pad_w, 0, pad_h), mode='reflect')
            hr_padded = F.pad(hr, (0, pad_w * scale, 0, pad_h * scale), mode='reflect')
        else:
            lr_padded = lr
            hr_padded = hr

        with torch.no_grad():
            if hasattr(self, "net_ema"):
                output = self.net_ema(lr_padded, hr_padded)
            else:
                output = self.net(lr_padded, hr_padded)
            preds = output.sample if hasattr(output, 'sample') else output
        if pad_h > 0 or pad_w > 0:
            preds = preds[..., :hr.size(2), :hr.size(3)]

        self.evaluator(preds, hr)

    def validation_step(self, batch: dict, batch_idx: int):
        self._shared_eval_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self):
        metrics = self.evaluator.compute()
        for k, v in metrics.items():
            # NOTE: FID is usually meaningless and too slow for small validation sets.
            # You can optionally skip logging FID during validation.
            if k == 'fid':
                continue
            # Disable prog_bar for epoch-level detailed metrics to avoid UI clutter
            self.log(f"val/{k}", v, prog_bar=True, sync_dist=True)
        self.evaluator.reset()

    def test_step(self, batch: dict, batch_idx:  int):
        self._shared_eval_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self):
        metrics = self.evaluator.compute()
        for k, v in metrics.items():
            self.log(f"test/{k}", v, prog_bar=False, sync_dist=True)
        save_dir = self.logger.log_dir
        save_filename = os.path.join(save_dir, "test_results_summary.xlsx")
        self.evaluator.save_to_excel(save_filename, metrics=metrics)
        self.evaluator.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_config)
        
        if self.lr_scheduler_config['type'] == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_scheduler_config['milestones'],
                gamma=self.lr_scheduler_config['gamma']
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            raise NotImplementedError(f"Scheduler {self.lr_scheduler_config['type']} is not implemented yet.")


    @torch.no_grad()
    def log_images(self, batch: dict, N: int = 4, **kwargs) -> dict:
        """
        Extracts image triplets (LR, Prediction, HR) and groups them by filename.
        Args:
            batch: The validation or training batch.
            N: Maximum number of images to log from the batch.
        """
        log = dict()
        
        # 1. Safely slice up to N images
        lr = batch['img'][:N]
        hr = batch['gt'][:N]
        
        # CRITICAL FIX: Find the ACTUAL number of images we got 
        # (It might be 2 even if N is 4, e.g., at the end of an epoch)
        actual_N = lr.shape[0]
        
        # 2. Extract or generate exactly 'actual_N' names to prevent IndexError
        img_names = batch.get('name', [f"image_{i}" for i in range(actual_N)])
        img_names = img_names[:actual_N]

                # Calculate padding needed to make LR dimensions multiples of 16
        _, _, h_lr, w_lr = lr.size()
        scale = hr.size(2) // h_lr
        pad_mult = 16
        pad_h = (pad_mult - h_lr % pad_mult) % pad_mult
        pad_w = (pad_mult - w_lr % pad_mult) % pad_mult
        # Apply 'reflect' padding if needed
        if pad_h > 0 or pad_w > 0:
            lr_padded = F.pad(lr, (0, pad_w, 0, pad_h), mode='reflect')
            hr_padded = F.pad(hr, (0, pad_w * scale, 0, pad_h * scale), mode='reflect')
        else:
            lr_padded = lr
            hr_padded = hr

        # 3. Get model predictions
        net_to_use = self.net_ema if hasattr(self, "net_ema") else self.net
        output = net_to_use(lr_padded, hr_padded)
        preds = output.sample if hasattr(output, 'sample') else output

        # 4. Crop predictions back to match original unpadded dimensions
        if pad_h > 0 or pad_w > 0:
            preds = preds[..., :hr.size(2), :hr.size(3)]
            
        # 4. Upsample LR image to match HR/SR dimensions
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False)

        # 5. Clamp strictly to [0, 1]
        lr_up = torch.clamp(lr_up, 0.0, 1.0)
        preds = torch.clamp(preds, 0.0, 1.0)
        hr = torch.clamp(hr, 0.0, 1.0)

        # 6. Group by image name. Loop will now safely run exactly 'actual_N' times.
        for i, name in enumerate(img_names):
            triplet = torch.stack([lr_up[i], preds[i], hr[i]], dim=0)
            log[name] = triplet

        return log