import importlib
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.metrics import SREvaluator
from utils import ENGINE_REGISTRY, build_network


def instantiate_from_config(config: dict):
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate from config.")
    
    module_path, class_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    
    params = config.get("params", {})
    return cls(**params)

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

        self.evaluator = SREvaluator(crop_border=eval_crop_border, test_y_channel=True)
        # self.net = instantiate_from_config(network_config)
        self.net = build_network(network_config)
        
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
        output = self.net(lr)
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
                output = self.net_ema(lr)
            else:
                output = self.net(lr)
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
        print(self.optimizer_config)
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