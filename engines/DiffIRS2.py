import importlib
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.metrics import SREvaluatorPyIQA
from torch import nn
from utils import ENGINE_REGISTRY, build_network
from .DiffIRS1 import DiffIRS1LightningModule
import os

class KDLoss(nn.Module):
    """
    Args:
        loss_weight (float): Loss weight for KD loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, temperature = 0.15):
        super(KDLoss, self).__init__()
    
        self.loss_weight = loss_weight
        self.temperature = temperature

    def forward(self, S1_fea, S2_fea):
        """
        Args:
            S1_fea (List): contain shape (N, L) vector. 
            S2_fea (List): contain shape (N, L) vector.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        loss_KD_dis = 0
        loss_KD_abs = 0
        for i in range(len(S1_fea)):
            S2_distance = F.log_softmax(S2_fea[i] / self.temperature, dim=1)
            S1_distance = F.softmax(S1_fea[i].detach()/ self.temperature, dim=1)
            loss_KD_dis += F.kl_div(
                        S2_distance, S1_distance, reduction='batchmean')
            loss_KD_abs += nn.L1Loss()(S2_fea[i], S1_fea[i].detach())
        return self.loss_weight * loss_KD_dis, self.loss_weight * loss_KD_abs
    

@ENGINE_REGISTRY.register()
class DiffIRS2LightningModule(pl.LightningModule):
    def __init__(
        self, 
        network_config_s2: dict,  # Config for the main S2 network
        network_config_s1: dict,  # Config for the frozen S1 network
        optimizer_config: dict,
        lr_scheduler_config: dict,
        train_config: dict,
        eval_crop_border: int = 4,
        img_size: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.automatic_optimization = False
                
        self.net = build_network(network_config_s2)
        self.net_ema = build_network(network_config_s2)

        s1_net = build_network(network_config_s1)
        self.model_Es1 = s1_net.E

        self.cri_pix = torch.nn.L1Loss() # Replace with your config-based instantiation
        self.cri_kd = KDLoss()  # Replace with your config-based instantiation
        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=True)

        s1_pretrain_path = train_config['pretrain_network_S1']
        if s1_pretrain_path:
            print(f"Loading pretrained weights from {s1_pretrain_path}...")
            ckpt = torch.load(s1_pretrain_path, map_location="cpu")
            if "params_ema" in ckpt:
                print(" ✅ Found 'params_ema'! Using EMA weights for initialization (Highly Recommended).")
                source_dict = ckpt["params_ema"]
            else:
                print(" ⚠️ 'params_ema' not found. Falling back to default 'state_dict'.")
                source_dict = ckpt["state_dict"]
            clean_dict = {}
            for k, v in source_dict.items():
                k_clean = k.replace("net_ema.", "").replace("net.", "")
                clean_dict[k_clean] = v
            es1_dict = {k.replace("E.", "", 1): v for k, v in clean_dict.items() if k.startswith("E.")}

            self.model_Es1.load_state_dict(es1_dict, strict=True)
            self.model_Es1.eval()
            self.model_Es1.requires_grad_(False)

            s2_init_dict = {}
            for k, v in clean_dict.items():
                if k.startswith("G."):
                    s2_init_dict[k] = v
                elif k.startswith("E."):
                    s2_init_dict[k.replace("E.", "condition.", 1)] = v 
            
            current_model_state = self.net.state_dict()
            keys_to_delete = []
            for k in s2_init_dict.keys():
                if k in current_model_state:
                    ckpt_shape = s2_init_dict[k].shape
                    model_shape = current_model_state[k].shape
                    
                    if ckpt_shape != model_shape:
                        print(f" ⚠️ Shape Mismatch [{k}]: ckpt {ckpt_shape} != model {model_shape}. Skipped (using random init).")
                        keys_to_delete.append(k)

            for k in keys_to_delete:
                del s2_init_dict[k]

            missing, unexpected = self.net.load_state_dict(s2_init_dict, strict=False)
            print(f"S2 Init: Missing keys (Should only be denoise/diffusion): {missing}")
            print(f"S2 Init: unexpected keys (Should only be denoise/diffusion): {unexpected}")
        

    def configure_optimizers(self):
        """
        Configure two distinct optimizers for the multi-stage training.
        """
        # Optimizer G: For the entire S2 network
        optim_params = [v for k, v in self.net.named_parameters() if v.requires_grad]
        optimizer = torch.optim.Adam(optim_params, **self.hparams.optimizer_config)
        
        if self.hparams.lr_scheduler_config['type'] == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.hparams.lr_scheduler_config['milestones'],
                gamma=self.hparams.lr_scheduler_config['gamma']
            )
        # Return as a list so Lightning registers both
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch: dict, batch_idx: int):
        # Access the manually configured optimizers
        opt = self.optimizers()
        
        hr = batch['gt']
        lr = batch['img']
        
        _, S1_IPR = self.model_Es1(lr, hr)


        opt.zero_grad()
        
        # Forward pass through the entire network
        output, pred_IPR_list = self.net(lr, S1_IPR[0])
        
        loss_pix = self.cri_pix(output, hr)
        
        S2_IPR = [pred_IPR_list[-1]]
        l_kd, l_abs = self.cri_kd(S1_IPR, S2_IPR)
        
        loss_total = loss_pix + l_abs
        
        self.manual_backward(loss_total)
        opt.step()
        
        self.log("train_loss_pix", loss_pix, prog_bar=True)
        self.log("train_loss_abs", l_abs, prog_bar=True)
        self.log("train_loss_kd", l_kd, prog_bar=True)

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        sch.step()

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        """
        Adapted for DiffIR S2: Includes Window Padding and Internalized Sampling.
        """
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

        # Select EMA model if available
        model = self.net_ema if hasattr(self, "net_ema") else self.net

        preds = model(lr_padded) 
        
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