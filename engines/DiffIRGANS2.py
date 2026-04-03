import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from utils import ENGINE_REGISTRY, build_network
from utils.metrics import SREvaluatorPyIQA
from basicsr.losses import build_loss
from basicsr.utils.registry import LOSS_REGISTRY
import os
from collections import OrderedDict


@LOSS_REGISTRY.register()
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
class DiffIRGANS2LightningModule(pl.LightningModule):
    def __init__(self, train_config, 
                 network_config_g, 
                 network_config_d, 
                 network_config_s1,
                 pretrain_s1_path: str,
                 pretrain_s2_path: str,
                 eval_crop_border: int = 4, 
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        # ==========================================
        # 0. CRITICAL: Disable automatic optimization! 
        # GANs require manual stepping to alternate G and D.
        # ==========================================
        self.automatic_optimization = False

        # ==========================================
        # 1. Initialize networks (Generator, Discriminator, S1 Teacher)
        # ==========================================
        self.net = build_network(network_config_g)
        self.net_d = build_network(network_config_d)
        
        # S1 acts solely as a Teacher to extract priors and is excluded from gradient updates
        self.net_g_S1 = build_network(network_config_s1)
        self.model_Es1 = self.net_g_S1.E # Assuming the feature extractor of S1 is named 'E'

        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=True)

        # ==========================================
        # 2. Load pretrained weights elegantly
        # ==========================================
        if pretrain_s1_path:
            self._load_network_weights(self.model_Es1, pretrain_s1_path, is_teacher=True)
            # Lock the teacher weights strictly
            self.model_Es1.eval()
            self.model_Es1.requires_grad_(False)

        if pretrain_s2_path:
            self._load_network_weights(self.net, pretrain_s2_path, is_teacher=False)

        # ==========================================
        # 3. Initialize loss functions
        # ==========================================
        loss_cfg = train_config.get('losses', {})
        self.cri_pix = build_loss(loss_cfg['pixel_opt']) if loss_cfg.get('pixel_opt') else None
        self.cri_kd = build_loss(loss_cfg['kd_opt']) if loss_cfg.get('kd_opt') else None
        self.cri_perceptual = build_loss(loss_cfg['perceptual_opt']) if loss_cfg.get('perceptual_opt') else None
        self.cri_gan = build_loss(loss_cfg['gan_opt']) if loss_cfg.get('gan_opt') else None

        self.net_d_iters = train_config.get('net_d_iters', 1)
        self.net_d_init_iters = train_config.get('net_d_init_iters', 0)

    def _load_network_weights(self, target_model: nn.Module, ckpt_path: str, is_teacher: bool = False):
        """
        A unified, elegant method to load checkpoint weights, prioritizing EMA weights,
        stripping PyTorch Lightning prefixes, and filtering specific sub-modules.
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
            
        print(f"==================================================")
        print(f"🚀 Loading weights from: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        # Prioritize EMA weights if they exist
        if "params_ema" in ckpt:
            print(" ✅ Source: 'params_ema' (EMA weights selected)")
            source_dict = ckpt["params_ema"]
        else:
            print(" ⚠️ Source: 'state_dict' (EMA not found, falling back)")
            source_dict = ckpt.get("state_dict", ckpt)
            
        clean_dict = OrderedDict()
        for k, v in source_dict.items():
            # Strip standard PyTorch Lightning prefixes
            k_clean = k.replace("net_ema.", "").replace("net.", "")
            
            # Apply Teacher (S1) specific filtering rules
            if is_teacher:
                if k_clean.startswith("E."):
                    k_clean = k_clean.replace("E.", "", 1)
                    clean_dict[k_clean] = v
            else:
                clean_dict[k_clean] = v

        missing_keys, unexpected_keys = target_model.load_state_dict(clean_dict, strict=True)
        if missing_keys:
            print(f" ⚠️ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f" ⚠️ Unexpected keys: {len(unexpected_keys)}")
        print(f"==================================================")


    def forward(self, lq):
        # Only use the Generator during inference
        return self.net(lq)

    def training_step(self, batch, batch_idx):
        # Retrieve manual optimizers
        opt_g, opt_d = self.optimizers()
        
        lq = batch['img']
        gt = batch['gt']
        current_iter = self.global_step

        opt_g.zero_grad()
        # ==========================================
        # Forward pass (Get Teacher prior and Generator output)
        # ==========================================
        with torch.no_grad():
            _, S1_IPR = self.model_Es1(lq, gt)
            
        # Note: If net_g returns (output, pred_IPR_list), ensure proper unpacking
        output, pred_IPR_list = self.net(lq, S1_IPR[0])

        # ==========================================
        # Phase 1: Optimize Generator (net_g)
        # ==========================================
        # toggle_optimizer automatically unfreezes G and freezes D!
        self.toggle_optimizer(opt_g) 
        
        l_g_total = 0
        log_dict = {}
        
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # 1. Pixel Loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(output, gt)
                l_g_total += l_g_pix
                log_dict['train/l_g_pix'] = l_g_pix
                
            # 2. Knowledge Distillation Loss
            if self.cri_kd:
                i = len(pred_IPR_list) - 1
                S2_IPR = [pred_IPR_list[i]]
                l_kd, l_abs = self.cri_kd(S1_IPR, S2_IPR)
                l_g_total += l_abs
                log_dict['train/l_kd'] = l_kd
                log_dict['train/l_abs'] = l_abs

            # 3. Perceptual Loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(output, gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    log_dict['train/l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    log_dict['train/l_g_style'] = l_g_style
                    
            # 4. GAN Loss (Generator fooling Discriminator)
            fake_g_pred = self.net_d(output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            log_dict['train/l_g_gan'] = l_g_gan

            # Execute backward pass and parameter update
            self.manual_backward(l_g_total)
            opt_g.step()
            
        self.untoggle_optimizer(opt_g)

        # ==========================================
        # Phase 2: Optimize Discriminator (net_d)
        # ==========================================
        self.toggle_optimizer(opt_d)
        
        # Real Loss
        real_d_pred = self.net_d(gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        
        # Fake Loss (Must detach to cut off gradient flow back to G!)
        fake_d_pred = self.net_d(output.detach().clone())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        
        l_d_total = l_d_real + l_d_fake
        
        self.manual_backward(l_d_total)
        opt_d.step()
        opt_d.zero_grad()
        
        self.untoggle_optimizer(opt_d)

        # ==========================================
        # Logging (Delegate uniformly to PL Logger)
        # ==========================================
        log_dict['train/l_d_real'] = l_d_real
        log_dict['train/l_d_fake'] = l_d_fake
        log_dict['train/out_d_real'] = torch.mean(real_d_pred.detach())
        log_dict['train/out_d_fake'] = torch.mean(fake_d_pred.detach())
        
        with torch.no_grad():
            preds_eval = torch.clamp(output.detach().float(), 0.0, 1.0)
            hr_eval = torch.clamp(gt.detach().float(), 0.0, 1.0)
            train_psnr = self.evaluator.psnr(preds_eval, hr_eval).mean().item()
            log_dict['train/psnr_epoch'] = train_psnr

        self.log_dict(log_dict, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        train_cfg = self.hparams.train_config
        lr_scheduler_config = self.hparams.train_config['lr_scheduler_config']

        opt_g = torch.optim.Adam(self.net.parameters(), **train_cfg['optim_g'])
        opt_d = torch.optim.Adam(self.net_d.parameters(), **train_cfg['optim_d'])

        sch_g = torch.optim.lr_scheduler.MultiStepLR(
            opt_g,
            milestones=lr_scheduler_config['milestones'],
            gamma=lr_scheduler_config['gamma']
        )
        
        sch_d = torch.optim.lr_scheduler.MultiStepLR(
            opt_d,
            milestones=lr_scheduler_config['milestones'],
            gamma=lr_scheduler_config['gamma']
        )


        return [
            {
                "optimizer": opt_g,
                "lr_scheduler": {
                    "scheduler": sch_g,
                    "interval": "epoch",
                    "frequency": 1,
                },
            },
            {
                "optimizer": opt_d,
                "lr_scheduler": {
                    "scheduler": sch_d,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        ]
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        sch_g, sch_d = self.lr_schedulers()
        
        sch_g.step()
        sch_d.step()

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        """
        Adapted for DiffIR S2: Includes Window Padding and Internalized Sampling.
        """
        hr = batch['gt']
        lr = batch['img']        

        # Select EMA model if available
        model = self.net_ema if hasattr(self, "net_ema") else self.net

        with torch.no_grad():
            output = model(lr)
            preds = output[0] if isinstance(output, tuple) else output
            
            preds_eval = torch.clamp(preds.float(), 0.0, 1.0)
            hr_eval = torch.clamp(hr.float(), 0.0, 1.0)

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