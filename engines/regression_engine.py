import importlib
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.metrics import SREvaluatorPyIQA
from utils import ENGINE_REGISTRY, build_network
import os


@ENGINE_REGISTRY.register()
class SRRegressionModule(pl.LightningModule):
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
        output = self.net(lr)
        # Handle diffusers UNet output wrapper if used, else use tensor directly
        preds = output.sample if hasattr(output, 'sample') else output
        
        # L1 Loss is mathematically strictly preferred over MSE for PSNR/SSIM optimization in standard SR
        loss = F.l1_loss(preds, hr)
        
        with torch.no_grad():
            preds_eval = torch.clamp(preds.detach().float(), 0.0, 1.0)
            hr_eval = torch.clamp(hr.detach().float(), 0.0, 1.0)
            train_psnr = self.evaluator.psnr(preds_eval, hr_eval).mean().item()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/psnr_epoch", train_psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        hr = batch['gt']
        lr = batch['img']
                
        with torch.no_grad():
            if hasattr(self, "net_ema"):
                output = self.net_ema(lr)
            else:
                output = self.net(lr)
            preds = output.sample if hasattr(output, 'sample') else output

        self.evaluator(preds, hr)

    def validation_step(self, batch: dict, batch_idx: int):
        self._shared_eval_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self):
        metrics = self.evaluator.compute()
        # Dynamically log all metrics to TensorBoard/Wandb
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
        print(self.optimizer_config)
        optimizer = torch.optim.Adam(self.net.parameters(), **self.optimizer_config)
        
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
    def inference(self, lr_tensor: torch.Tensor) -> torch.Tensor:
        model = self.net_ema if hasattr(self, "net_ema") else self.net
        
        
        _, _, h_old, w_old = lr_tensor.size()
        h_pad = (8 - h_old % 8) % 8
        w_pad = (8 - w_old % 8) % 8
        if h_pad or w_pad:
            lr_tensor_padded = F.pad(lr_tensor, (0, w_pad, 0, h_pad), mode='reflect')

            preds = model(lr_tensor_padded)   

            scale = getattr(self.hparams, 'eval_crop_border', 4)
            h_hr_true = h_old * scale
            w_hr_true = w_old * scale

            preds = preds[..., :h_hr_true, :w_hr_true]
        else:
            preds = model(lr_tensor)

        return torch.clamp(preds, 0.0, 1.0)

    @torch.no_grad()
    def log_images(self, batch: dict, N: int = 4, **kwargs) -> dict:
        """
        Extracts image triplets (LR, Prediction, HR) and groups them by filename.
        Args:
            batch: The validation or training batch.
            N: Maximum number of images to log from the batch.
        """
        log = dict()
        
        lr = batch['img'][:N]
        hr = batch['gt'][:N]
        actual_N = lr.shape[0]

        # 1. Safely extract image filenames. If dataloader doesn't provide 'name', use fallbacks.
        img_names = batch.get('name', [f"image_{i}" for i in range(actual_N)])
        img_names = img_names[:N]

        # 2. Get model predictions
        net_to_use = self.net_ema if hasattr(self, "net_ema") else self.net
        output = net_to_use(lr)
        preds = output.sample if hasattr(output, 'sample') else output

        # 3. Upsample LR image to match HR/SR dimensions
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False)

        # 4. Clamp strictly to [0, 1]
        lr_up = torch.clamp(lr_up, 0.0, 1.0)
        preds = torch.clamp(preds, 0.0, 1.0)
        hr = torch.clamp(hr, 0.0, 1.0)

        # 5. Group by image name. 
        # We create a triplet tensor of shape (3, C, H, W) for each image.
        for i, name in enumerate(img_names):
            # The order here defines how they appear from left to right: [LR, Prediction, HR]
            triplet = torch.stack([lr_up[i], preds[i], hr[i]], dim=0)
            log[name] = triplet

        return log