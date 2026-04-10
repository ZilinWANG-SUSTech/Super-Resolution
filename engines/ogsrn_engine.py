import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict
from utils.metrics import SREvaluatorPyIQA
from utils import ENGINE_REGISTRY, build_network


@ENGINE_REGISTRY.register()
class OGSRNModule(pl.LightningModule):
    def __init__(
        self,
        network_config: dict,
        sortn_config: dict,
        optimizer_config: dict,
        pretrain_sortn_path: str,
        lr_scheduler_config: dict = None,
        eval_loss_weight: float = 0.1,    # lambda_var in train(2).py
        resolution_method: str = 'Single Pass', # 'Single Pass' or 'Two Step'
        eval_crop_border: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.lambda_var = eval_loss_weight
        self.res_method = resolution_method

        # Initialize Evaluator (for SAR PSNR/SSIM)
        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=True)
        
        # Build main SRUN network
        self.net = build_network(network_config)
        
        # Build guidance SORTN network
        self.net_sortn = build_network(sortn_config)
        
        # Load pre-trained SORTN weights
        self._load_sortn_weights(pretrain_sortn_path)
        
        # Freeze SORTN: It acts as a fixed teacher/evaluator in Stage-2
        self.net_sortn.eval()
        for param in self.net_sortn.parameters():
            param.requires_grad = False

    def _load_sortn_weights(self, path):
    # Load checkpoint to CPU memory first to avoid GPU fragmentation
        checkpoint = torch.load(path, map_location='cpu')
        
        # Extract the full state_dict (Lightning puts it under 'state_dict' key)
        full_state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Target state_dict for self.net_sortn
        new_state_dict = OrderedDict()
        
        # Define the prefixes we want to keep (Generator weights)
        # In your previous SORTN training, the generator was likely named 'net_g'
        target_prefixes = ['net_g.', 'net.'] 
        
        # Define keywords to strictly ignore (Metrics, Evaluators, etc.)
        ignore_keywords = ['evaluator', 'discriminator', 'optimizer', 'scheduler']

        for k, v in full_state_dict.items():
            # Check if the key belongs to an ignored module
            if any(ik in k for ik in ignore_keywords):
                continue
                
            # Check if the key matches our desired generator prefixes
            for prefix in target_prefixes:
                if k.startswith(prefix):
                    # Remove the prefix to match local nn.Module's keys
                    # e.g., 'net_g.blocks.0.weight' -> 'blocks.0.weight'
                    new_key = k[len(prefix):]
                    new_state_dict[new_key] = v
                    break

        if not new_state_dict:
            raise ValueError(f"❌ Could not find any valid generator weights in {path}. Check your checkpoint keys.")

        # Load with strict=False to avoid crashing on minor architecture differences
        missing_keys, unexpected_keys = self.net_sortn.load_state_dict(new_state_dict, strict=False)
        
        print(f"==================================================")
        print(f"🚀 Pre-trained SORTN weights filtered from: {path}")
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys ignored: {len(unexpected_keys)}")
        print(f"==================================================")

    def forward(self, lr_img: torch.Tensor, res_label: torch.Tensor = None) -> torch.Tensor:
        if self.res_method == 'Two Step':
            return self.net(lr_img, res_label)
        return self.net(lr_img)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        optic_gt = batch['guide']   # Real optical image
        sar_hr = batch['hr']        # High-res SAR ground truth
        sar_lr = batch['lr']       # Low-res SAR input
        res_label = batch.get('res_label', None)
        
        # 1. Forward pass: Reconstruct SR SAR image
        if self.res_method == 'Two Step':
            output = self.net(sar_lr, res_label)
        else:
            output = self.net(sar_lr)
        
        # Handle tuple output (x, f_maps) from your SRUN implementation
        sar_sr = output[0] if isinstance(output, tuple) else output

        # 2. Content Loss (SAR Space L1 Loss)
        content_loss = F.l1_loss(sar_sr, sar_hr)

        # 3. Evaluation Loss (Optical Space Feedback)
        # We use SORTN to compute losses in optical space without updating SORTN
        with torch.no_grad():
            # Interpolate both to 256x256 to fit SORTN input requirements
            # sar_hr_256 = F.interpolate(sar_hr, size=[256, 256], mode='bicubic', align_corners=False)
            # SORTN returns (optic_img, f_maps)
            optic_gen_hr = self.net_sortn(sar_hr)[0]
            l_hr = F.l1_loss(optic_gen_hr, optic_gt)

        # sar_sr_256 = F.interpolate(sar_sr, size=[256, 256], mode='bicubic', align_corners=False)
        optic_gen_sr = self.net_sortn(sar_sr)[0]
        l_sr = F.l1_loss(optic_gen_sr, optic_gt)

        # Feedback information: minimize difference between SR and HR behavior in optical space
        evaluation_loss = F.l1_loss(l_hr, l_sr)

        # 4. Combined Loss
        total_loss = content_loss + self.lambda_var * evaluation_loss

        # Logging
        self.log("train/total_loss", total_loss, prog_bar=True)
        self.log("train/content_loss", content_loss)
        self.log("train/eval_loss", evaluation_loss)
        
        return total_loss

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        sar_hr = batch['hr']
        sar_lr = batch['lr']
        res_label = batch.get('res_label', None)
                
        # Use EMA if available, otherwise use standard net
        model = self.net_ema if hasattr(self, "net_ema") else self.net
        
        if self.res_method == 'Two Step':
            output = model(sar_lr, res_label)
        else:
            output = model(sar_lr)
            
        sar_sr = output[0] if isinstance(output, tuple) else output
        
        # Standard SAR metrics evaluation
        preds_eval = torch.clamp(sar_sr.detach().float(), 0.0, 1.0)
        hr_eval = torch.clamp(sar_hr.detach().float(), 0.0, 1.0)
        self.evaluator(preds_eval, hr_eval)

    def validation_step(self, batch: dict, batch_idx: int):
        self._shared_eval_step(batch, batch_idx, stage="val")

    def on_validation_epoch_end(self):
        metrics = self.evaluator.compute()
        for k, v in metrics.items():
            if k == 'fid': continue
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
    def inference(self, lr_tensor: torch.Tensor, guided_tensor: torch.Tensor = None, res_label: torch.Tensor = None) -> torch.Tensor:
        model = self.net_ema if hasattr(self, "net_ema") else self.net
        
        # Ensure stride factor 16 as discussed for SORTN/SRUN architectures
        stride_factor = 16 
        _, _, h_old, w_old = lr_tensor.size()
        h_pad = (stride_factor - h_old % stride_factor) % stride_factor
        w_pad = (stride_factor - w_old % stride_factor) % stride_factor
        
        if h_pad or w_pad:
            lr_tensor = F.pad(lr_tensor, (0, w_pad, 0, h_pad), mode='reflect')

        if self.res_method == 'Two Step':
            output = model(lr_tensor, res_label)
        else:
            output = model(lr_tensor)
            
        preds = output[0] if isinstance(output, tuple) else output

        # Crop back to original scale
        scale = self.hparams.get('scale_factor', 4) # Assuming scale_factor is in hparams
        h_hr_true, w_hr_true = h_old * scale, w_old * scale
        preds = preds[..., :h_hr_true, :w_hr_true]
        
        return torch.clamp(preds, 0.0, 1.0)