import importlib
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers import DDPMScheduler, DDIMScheduler
from utils.metrics import SREvaluatorPyIQA
from tqdm import tqdm
from utils import ENGINE_REGISTRY


def instantiate_from_config(config: dict):
    """
    Dynamically instantiates a class from a configuration dictionary.
    """
    if "target" not in config:
        raise KeyError("Expected key 'target' to instantiate from config.")
    
    module_path, class_name = config["target"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    
    params = config.get("params", {})
    return cls(**params)


@ENGINE_REGISTRY.register()
class SRDiffusionModule(pl.LightningModule):
    def __init__(
        self, 
        network_config: dict,
        optimizer_config: dict,
        diffusion_process_config: dict,
        eval_crop_border: int = 4, # Exposed for YAML configuration
        lr_scheduler_config: dict = None,
        img_size: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config

        # Initialize the new SwinIR-style Evaluator
        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=True)
        
        self.net = instantiate_from_config(network_config)
        
        self.diffusion_process = instantiate_from_config(diffusion_process_config)

        # self.example_input_array = (torch.randn(1, 6, 32, 32), torch.randint(0, 1000, (1,)))
    
    def forward(self, lr_img: torch.Tensor) -> torch.Tensor:
        """Inference only forward pass."""
        pass

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        hr = batch['gt'] * 2.0 - 1.0
        lr = batch['img'] * 2.0 - 1.0
        bsz = hr.shape[0]
        
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False)
        
        loss = self.diffusion_process.compute_training_loss(
            net=self.net, 
            x_start=hr, 
            cond=lr_up
        )

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        """
        Shared reverse diffusion logic for both validation and testing.
        """
        hr = batch['gt'] * 2.0 - 1.0
        lr = batch['img'] * 2.0 - 1.0

        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False)
        
        net_to_use = self.net_ema if hasattr(self, "net_ema") else self.net        
        preds = self.diffusion_process.sample(
            net=net_to_use,
            shape=hr.shape,
            device=self.device,
            cond=lr_up
        )

        preds_01 = torch.clamp((preds + 1.0) / 2.0, 0.0, 1.0)
        hr_01 = torch.clamp((hr + 1.0) / 2.0, 0.0, 1.0)
        
        # Calculates metrics and automatically updates internal states
        metrics = self.evaluator(preds_01, hr_01)

        # Log per-batch metrics
        self.log(f"{stage}/psnr_step", metrics['psnr'], prog_bar=True, sync_dist=True)
        self.log(f"{stage}/ssim_step", metrics['ssim'], prog_bar=True, sync_dist=True)
    
    def validation_step(self, batch: dict, batch_idx: int):
        self._shared_eval_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self):
        """Calculates global average and safely resets states."""
        metrics = self.evaluator.compute()

        # Dynamically log all metrics to TensorBoard/Wandb
        for k, v in metrics.items():
            # NOTE: FID is usually meaningless and too slow for small validation sets.
            # You can optionally skip logging FID during validation.
            if k == 'fid':
                continue
            # Disable prog_bar for epoch-level detailed metrics to avoid UI clutter
            self.log(f"val/{k}", v, prog_bar=False, sync_dist=True)

        # Unified reset to clear states for the next epoch
        self.evaluator.reset()

    def test_step(self, batch: dict, batch_idx: int):
        self._shared_eval_step(batch, batch_idx, stage="test")

    def on_test_epoch_end(self):
        """Calculates global average and safely resets states."""
        metrics = self.evaluator.compute()

        for k, v in metrics.items():
            self.log(f"test/{k}", v, prog_bar=False, sync_dist=True)

        self.evaluator.reset()

    def configure_optimizers(self):
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
        elif self.lr_scheduler_config['type'] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.lr_scheduler_config['T_max'],
                eta_min=self.lr_scheduler_config.get('eta_min', 1e-7)
            )
        else:
            raise NotImplementedError(f"Scheduler {self.lr_scheduler_config['type']} is not implemented yet.")