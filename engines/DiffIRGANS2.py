import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from utils import ENGINE_REGISTRY, build_network
from utils.metrics import SREvaluatorPyIQA
from basicsr.losses import build_loss
from basicsr.utils.registry import LOSS_REGISTRY


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
    def __init__(self, train_config, network_config_g, network_config_d, network_config_s1, eval_crop_border: int = 4, mg_size: int = None,):
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
        self.net_ema = build_network(network_config_g)
        self.net_d = build_network(network_config_d)
        
        # S1 acts solely as a Teacher to extract priors and is excluded from gradient updates
        self.net_g_S1 = build_network(network_config_s1)
        self.model_Es1 = self.net_g_S1.E # Assuming the feature extractor of S1 is named 'E'

        self.evaluator = SREvaluator(crop_border=eval_crop_border, test_y_channel=True)

        # ==========================================
        # 2. Load pretrained weights (using our bulletproof logic)
        # ==========================================
        self._load_pretrained_weights()

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

    def _load_pretrained_weights(self):
        """
        Reuse the EMA extraction and Shape interception logic we refined earlier.
        Omitted here for brevity. Please paste the S1 and S2 pretrained loading code directly here.
        Also remember: freeze self.model_Es1 completely!
        """
        s1_pretrain_path = self.hparams.train_config['pretrain_network_S1']
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
        
        s2_pretrain_path = self.hparams.train_config['pretrain_network_S2']
        if s2_pretrain_path:
            print(f"Loading pretrained weights from {s2_pretrain_path}...")
            ckpt = torch.load(s2_pretrain_path, map_location="cpu")
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

            self.net.load_state_dict(clean_dict, strict=True)


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

        preds = model(lr) 
        
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