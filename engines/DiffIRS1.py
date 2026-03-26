import importlib
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.metrics import SREvaluator
from utils import ENGINE_REGISTRY


def instantiate_from_config(config: dict):
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate from config.")
    
    module_path, class_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    
    params = config.get("params", {})
    return cls(**params)

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

        self.evaluator = SREvaluator(crop_border=eval_crop_border, test_y_channel=True)
        self.net = instantiate_from_config(network_config)
        
        # img_size = network_config['params']['img_size']
        # self.example_input_array = torch.randn(1, 3, img_size, img_size)
        # self.example_input_array = (torch.randn(1, 3, img_size, img_size),torch.randn(1, 3, img_size, img_size))

    def forward(self, lr_img: torch.Tensor) -> torch.Tensor:
        # Directly output HR prediction
        # return self.net(lr_img, gt_img)
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
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        hr = batch['gt']
        lr = batch['img']
                
        with torch.no_grad():
            if hasattr(self, "net_ema"):
                output = self.net_ema(lr, hr)
            else:
                output = self.net(lr, hr)
            preds = output.sample if hasattr(output, 'sample') else output

        metrics = self.evaluator(preds, hr)
        self.log(f"{stage}_psnr_step", metrics['psnr'], prog_bar=True, sync_dist=True)
        self.log(f"{stage}_ssim_step", metrics['ssim'], prog_bar=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int):
        self._shared_eval_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self):
        metrics = self.evaluator.compute()
        self.log("val_psnr", metrics['psnr'], prog_bar=True, sync_dist=True)
        self.log("val_ssim", metrics['ssim'], prog_bar=True, sync_dist=True)
        self.evaluator.reset()

    def test_step(self, batch: dict, batch_idx:  int):
        self._shared_eval_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self):
        metrics = self.evaluator.compute()
        self.log("test_psnr", metrics['psnr'], prog_bar=True, sync_dist=True)
        self.log("test_ssim", metrics['ssim'], prog_bar=True, sync_dist=True)
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