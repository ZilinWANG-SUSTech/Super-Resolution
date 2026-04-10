import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# Assuming you have these imported from your project structure
from utils.metrics import SREvaluatorPyIQA
from utils import ENGINE_REGISTRY, build_network

# Import the UGSR generator we built previously
# from ugsr_network import UGSRGenerator 

def image_gradients(image):
    """
    Computes gradients along the Height (dy) and Width (dx) dimensions.
    """
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1))
    
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dx = F.pad(dx, (0, 1, 0, 0))
    
    return dy, dx

def gradient_loss(output, gt):
    """
    Computes the L1 difference between the gradients of output and gt.
    """
    dy_out, dx_out = image_gradients(output)
    dy_gt, dx_gt = image_gradients(gt)
    
    loss_y = F.l1_loss(dy_out, dy_gt, reduction='mean')
    loss_x = F.l1_loss(dx_out, dx_gt, reduction='mean')
    
    return (loss_y + loss_x) / 2.0


@ENGINE_REGISTRY.register()
class UGSRLightningModule(pl.LightningModule):
    def __init__(
        self, 
        network_config: dict,
        optimizer_config: dict,
        eval_crop_border: int = 4, 
        lr_scheduler_config: dict = None,
        img_size: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config

        # Initialize the evaluator
        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=True)
        
        # Build the network (Assuming build_network handles your UGSRGenerator)
        self.net = build_network(network_config)
        
        # Loss weights based on original TF implementation
        self.weight_rec = 10.0
        self.weight_grad = 2.0

    def forward(self, lr_img: torch.Tensor, guide_img: torch.Tensor) -> torch.Tensor:
        return self.net(lr_img, guide_img)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Extract the multi-modal data
        lr = batch['lr']
        hr = batch['hr']
        guide = batch['guide']
        
        # Forward pass
        preds = self(lr, guide)
        
        # Calculate losses (in [-1, 1] space, which is mathematically fine)
        l1_loss = F.l1_loss(preds, hr)
        grad_loss = gradient_loss(preds, hr)
        
        total_loss = self.weight_rec * l1_loss + self.weight_grad * grad_loss
        
        with torch.no_grad():
            # Denormalize from [-1, 1] to [0, 1] for metrics calculation
            preds_eval = torch.clamp((preds.detach() + 1.0) / 2.0, 0.0, 1.0)
            hr_eval = torch.clamp((hr.detach() + 1.0) / 2.0, 0.0, 1.0)
            train_psnr = self.evaluator.psnr(preds_eval, hr_eval).mean().item()
            
        self.log("train_loss", total_loss, prog_bar=True, sync_dist=True)
        self.log("train_l1_loss", l1_loss, prog_bar=False, sync_dist=True)
        self.log("train_grad_loss", grad_loss, prog_bar=False, sync_dist=True)
        self.log("train/psnr_epoch", train_psnr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return total_loss

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        lr = batch['lr']
        hr = batch['hr']
        guide = batch['guide']
                
        with torch.no_grad():
            output = self.net_ema(lr, guide) if hasattr(self, "net_ema") else self(lr, guide)
            
            # Denormalize from [-1, 1] to [0, 1] for evaluation
            preds_eval = torch.clamp((output.detach() + 1.0) / 2.0, 0.0, 1.0)
            hr_eval = torch.clamp((hr.detach() + 1.0) / 2.0, 0.0, 1.0)
            
        self.evaluator(preds_eval, hr_eval)

    def validation_step(self, batch: dict, batch_idx: int):
        self._shared_eval_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self):
        metrics = self.evaluator.compute()
        for k, v in metrics.items():
            if k == 'fid':
                continue
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
    def log_images(self, batch: dict, N: int = 4, **kwargs) -> dict:
        """
        Extracts image quadruplets (LR, Guide, Prediction, HR) and groups them.
        """
        log = dict()
        
        lr = batch['lr'][:N]
        hr = batch['hr'][:N]
        guide = batch['guide'][:N]
        actual_N = lr.shape[0]

        img_names = batch.get('img_name', [f"image_{i}" for i in range(actual_N)])
        img_names = img_names[:N]

        net_to_use = self.net_ema if hasattr(self, "net_ema") else self.net
        preds = net_to_use(lr, guide)

        # Upsample LR image for visual comparison
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False)

        # Denormalize all tensors from [-1, 1] back to [0, 1] for visualization
        lr_up = torch.clamp((lr_up + 1.0) / 2.0, 0.0, 1.0)
        guide = torch.clamp((guide + 1.0) / 2.0, 0.0, 1.0)
        preds = torch.clamp((preds + 1.0) / 2.0, 0.0, 1.0)
        hr = torch.clamp((hr + 1.0) / 2.0, 0.0, 1.0)

        for i, name in enumerate(img_names):
            # The order defines how they appear from left to right: [LR, Guide, Prediction, HR]
            quadruplet = torch.stack([lr_up[i], guide[i], preds[i], hr[i]], dim=0)
            log[name] = quadruplet

        return log