import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict
from utils.metrics import SREvaluatorPyIQA
from utils import ENGINE_REGISTRY, build_network
from basicsr.losses import build_loss


@ENGINE_REGISTRY.register()
class SRGANModule(pl.LightningModule):
    def __init__(
        self,
        network_g_config: dict,
        network_d_config: dict,
        optimizer_g_config: dict,
        optimizer_d_config: dict,
        losses_config: dict,  # NEW: Pass the 'train' dict from YAML that contains all *_opt
        eval_crop_border: int = 4,
        lr_scheduler_g_config: dict = None,
        lr_scheduler_d_config: dict = None,
        pretrain_g_path: str = None,  # NEW: Path to the pre-trained regression checkpoint
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Disable automatic optimization for fine-grained control over GAN training
        self.automatic_optimization = False

        self.optimizer_g_config = optimizer_g_config
        self.optimizer_d_config = optimizer_d_config
        self.lr_scheduler_g_config = lr_scheduler_g_config
        self.lr_scheduler_d_config = lr_scheduler_d_config

        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=True)
        
        # Build networks
        self.net = build_network(network_g_config)
        self.net_d = build_network(network_d_config)
        
        # Initialize losses dynamically from YAML config
        self.cri_pix = build_loss(losses_config['pixel_opt']) if losses_config.get('pixel_opt') else None
        self.cri_perceptual = build_loss(losses_config['perceptual_opt']) if losses_config.get('perceptual_opt') else None
        self.cri_gan = build_loss(losses_config['gan_opt']) if losses_config.get('gan_opt') else None

        # Load pre-trained weights for the generator if provided
        if pretrain_g_path is not None:
            self._load_pretrained_generator(pretrain_g_path)

    def _load_pretrained_generator(self, ckpt_path: str) -> None:
        """
        Parses a PyTorch Lightning checkpoint from a Regression phase and loads 
        the optimal weights (prioritizing EMA) into the current GAN generator.
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Pre-trained checkpoint not found at: {ckpt_path}")
            
        # Load the checkpoint dictionary onto CPU to prevent VRAM spikes
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Dictionary to hold the filtered and renamed weights for net_g
        g_state_dict = OrderedDict()
        
        # Check if EMA weights exist in the loaded checkpoint
        has_ema = any(k.startswith('net_ema.') for k in state_dict.keys())
        prefix_to_search = 'net_ema.' if has_ema else 'net.'
        
        for k, v in state_dict.items():
            if k.startswith(prefix_to_search):
                # Strip the prefix ('net_ema.' or 'net.') to match the raw nn.Module keys
                # because we will use self.net_g.load_state_dict() directly.
                new_key = k.replace(prefix_to_search, '')
                g_state_dict[new_key] = v
                
        # Fallback: if the checkpoint is a raw PyTorch state dict without PL wrappers
        if len(g_state_dict) == 0:
            g_state_dict = state_dict
            prefix_to_search = 'RAW PyTorch Dict'
            
        # Load the mapped weights into the generator
        # strict=False allows loading even if there are slight architecture mismatches 
        # (e.g., if you added extra scale-up layers later)
        missing_keys, unexpected_keys = self.net.load_state_dict(g_state_dict, strict=False)
        
        print(f"==================================================")
        print(f"🚀 Generator weights initialized from: {ckpt_path}")
        print(f"🔍 Weight source used: {prefix_to_search}")
        if missing_keys:
            print(f"⚠️ Missing keys (expected but not found): {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️ Unexpected keys (found but not needed): {len(unexpected_keys)}")
        print(f"==================================================")

    def forward(self, lr_img: torch.Tensor) -> torch.Tensor:
        return self.net(lr_img)

    def training_step(self, batch: dict, batch_idx: int) -> None:
        hr = batch['gt']
        lr = batch['img']
        
        opt_g, opt_d = self.optimizers()

        # =========================================================
        # 1. Train Generator
        # =========================================================
        self.toggle_optimizer(opt_g)
        
        preds = self.net(lr)
        if hasattr(preds, 'sample'):
            preds = preds.sample
            
        loss_g = 0.0
        log_dict_g = {}
        
        # Pixel Loss
        if self.cri_pix:
            l_g_pix = self.cri_pix(preds, hr)
            loss_g += l_g_pix
            log_dict_g["train/g_pix"] = l_g_pix

        # Perceptual & Style Loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(preds, hr)
            if l_g_percep is not None:
                loss_g += l_g_percep
                log_dict_g["train/g_percep"] = l_g_percep
            if l_g_style is not None:
                loss_g += l_g_style
                log_dict_g["train/g_style"] = l_g_style
                
        # GAN Loss for Generator (target_is_real=True, is_disc=False)
        if self.cri_gan:
            fake_g_pred = self.net_d(preds)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            loss_g += l_g_gan
            log_dict_g["train/g_gan"] = l_g_gan

        log_dict_g["train/g_total"] = loss_g
        
        self.manual_backward(loss_g)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # =========================================================
        # 2. Train Discriminator
        # =========================================================
        self.toggle_optimizer(opt_d)
        
        loss_d = 0.0
        log_dict_d = {}

        if self.cri_gan:
            # Real Loss (target_is_real=True, is_disc=True)
            real_d_pred = self.net_d(hr)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            log_dict_d["train/d_real"] = l_d_real
            loss_d += l_d_real
            
            # Fake Loss (target_is_real=False, is_disc=True)
            # Detach to prevent gradients flowing back into Generator
            fake_d_pred = self.net_d(preds.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            log_dict_d["train/d_fake"] = l_d_fake
            loss_d += l_d_fake

        log_dict_d["train/d_total"] = loss_d

        # Only backward and step if GAN loss is actually configured
        if self.cri_gan:
            self.manual_backward(loss_d)
            opt_d.step()
            opt_d.zero_grad()
            
        self.untoggle_optimizer(opt_d)

        # =========================================================
        # 3. Logging & Metrics
        # =========================================================
        with torch.no_grad():
            preds_eval = torch.clamp(preds.detach().float(), 0.0, 1.0)
            hr_eval = torch.clamp(hr.detach().float(), 0.0, 1.0)
            train_psnr = self.evaluator.psnr(preds_eval, hr_eval).mean().item()
            
        # Combine logs and push to logger
        full_log = {**log_dict_g, **log_dict_d, "train/psnr_epoch": train_psnr}
        self.log_dict(full_log, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)


    def on_train_batch_end(self, outputs, batch, batch_idx):
        sch_g, sch_d = self.lr_schedulers()
        
        sch_g.step()
        sch_d.step()


    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        hr = batch['gt']
        lr = batch['img']
                
        with torch.no_grad():
            if hasattr(self, "net_ema"):
                output = self.net_ema(lr)
            else:
                output = self.net(lr)
            preds = output.sample if hasattr(output, 'sample') else output
            
            # Ensure float32 casting here as well for safety
            preds_eval = torch.clamp(preds.float(), 0.0, 1.0)
            hr_eval = torch.clamp(hr.float(), 0.0, 1.0)

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

    def test_step(self, batch: dict, batch_idx: int):
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
        opt_g = torch.optim.Adam(self.net.parameters(), **self.optimizer_g_config)
        opt_d = torch.optim.Adam(self.net_d.parameters(), **self.optimizer_d_config)
        
        # Configure Scheduler for Generator
        if self.lr_scheduler_g_config['type'] == "MultiStepLR":
            sch_g = torch.optim.lr_scheduler.MultiStepLR(
                opt_g,
                milestones=self.lr_scheduler_g_config['milestones'],
                gamma=self.lr_scheduler_g_config['gamma']
            )
        else:
            raise NotImplementedError(f"G Scheduler not implemented.")
            
        # Configure Scheduler for Discriminator
        if self.lr_scheduler_d_config['type'] == "MultiStepLR":
            sch_d = torch.optim.lr_scheduler.MultiStepLR(
                opt_d,
                milestones=self.lr_scheduler_d_config['milestones'],
                gamma=self.lr_scheduler_d_config['gamma']
            )
        else:
            raise NotImplementedError(f"D Scheduler not implemented.")

        return [opt_g, opt_d], [sch_g, sch_d]

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
        log = dict()
        lr = batch['img'][:N]
        hr = batch['gt'][:N]
        actual_N = lr.shape[0]

        img_names = batch.get('name', [f"image_{i}" for i in range(actual_N)])[:N]

        net_to_use = self.net_ema if hasattr(self, "net_ema") else self.net
        output = net_to_use(lr)
        preds = output.sample if hasattr(output, 'sample') else output

        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False)

        lr_up = torch.clamp(lr_up, 0.0, 1.0)
        preds = torch.clamp(preds, 0.0, 1.0)
        hr = torch.clamp(hr, 0.0, 1.0)

        for i, name in enumerate(img_names):
            triplet = torch.stack([lr_up[i], preds[i], hr[i]], dim=0)
            log[name] = triplet

        return log