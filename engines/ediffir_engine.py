import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import einops

# Import your custom modules based on your project structure
from utils.metrics import SREvaluatorPyIQA
from utils import ENGINE_REGISTRY, build_network
from models import IRSDE
from tqdm import tqdm

class MatchingLoss(nn.Module):
    def __init__(self, loss_type='l1', is_weighted=False):
        super().__init__()
        self.is_weighted = is_weighted

        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        elif loss_type == 'l2':
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def forward(self, predict, target, weights=None):

        loss = self.loss_fn(predict, target, reduction='none')
        loss = einops.reduce(loss, 'b ... -> b (...)', 'mean')

        if self.is_weighted and weights is not None:
            loss = weights * loss

        return loss.mean()

@ENGINE_REGISTRY.register()
class IRSDEEngine(pl.LightningModule):
    def __init__(
        self, 
        network_config: dict,
        sde_config: dict,       
        optimizer_config: dict,
        loss_config: dict,      
        eval_crop_border: int = 4, 
        scale_factor: int = 4,
        lr_scheduler_config: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.scale_factor = scale_factor

        # 1. Evaluator
        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=True)
        
        # 2. Build Network
        self.net = build_network(network_config)

        # 3. Build IRSDE
        self.sde = IRSDE(
            max_sigma=sde_config["max_sigma"], 
            T=sde_config["T"], 
            schedule=sde_config["schedule"], 
            eps=sde_config["eps"], 
            device="cuda"
        )
        
        # 4. Define Loss
        self.loss_fn = MatchingLoss(
            loss_type=loss_config.get('loss_type', 'l1'), 
            is_weighted=loss_config.get('is_weighted', False)
        )
        self.loss_weight = loss_config.get('weight', 1.0)

    def _upscale_lr(self, lr: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """Upscale LR to match GT dimensions (Bicubic interpolation)"""
        return F.interpolate(lr, size=target_shape[-2:], mode='bicubic', align_corners=False)

    def forward(self, lr_img: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        hr = batch['gt']
        lr = batch['img']
        
        # Upscale LR to HR dimension as mu (mu and GT must have the same shape in IR-SDE)
        mu = self._upscale_lr(lr, hr.shape)
        
        # Set the current training network to SDE
        self.sde.set_model(self.net)
        
        # Generate random timesteps and noisy states using x0(GT) and mu(upscaled LQ)
        timesteps, noisy_states = self.sde.generate_random_states(x0=hr, mu=mu)
        timesteps = timesteps.to(self.device)

        # Set mu to the conditional upscaled image
        self.sde.set_mu(mu)

        # Get noise and score from the network
        noise = self.sde.noise_fn(noisy_states, timesteps.squeeze())
        score = self.sde.get_score_from_noise(noise, timesteps)

        # Learning the maximum likelihood objective for state x_{t-1}
        xt_1_expection = self.sde.reverse_sde_step_mean(noisy_states, score, timesteps)
        xt_1_optimum = self.sde.reverse_optimum_step(noisy_states, hr, timesteps)
        
        # Calculate matching loss
        loss = self.loss_weight * self.loss_fn(xt_1_expection, xt_1_optimum)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def _get_eval_model(self):
        """Safely fetch the EMA model if external callback injected it, else fallback to training net."""
        return self.net_ema if hasattr(self, "net_ema") else self.net

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        hr = batch['gt']
        lr = batch['img']
        
        mu = self._upscale_lr(lr, hr.shape)
                
        with torch.no_grad():
            net_to_use = self._get_eval_model()
            self.sde.set_model(net_to_use)
            self.sde.set_mu(mu)
            
            # Generate initial noisy state (x_T) based on upscaled LR image
            xt = self.sde.noise_state(mu)
            
            # Run the reverse SDE process to get the prediction
            preds = self.sde.reverse_sde(xt)
            
            # Clamp for evaluation
            preds_eval = torch.clamp(preds.detach().float(), 0.0, 1.0)
            hr_eval = torch.clamp(hr.detach().float(), 0.0, 1.0)
            
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
        opt_type = self.optimizer_config.pop('type', 'Adam')
        
        if opt_type == 'Adam':
            optimizer = torch.optim.Adam(self.net.parameters(), **self.optimizer_config)
        elif opt_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.net.parameters(), **self.optimizer_config)
        else:
            optimizer = torch.optim.Adam(self.net.parameters(), **self.optimizer_config)
        
        if self.lr_scheduler_config['type'] == "MultiStepLR":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.lr_scheduler_config['milestones'],
                gamma=self.lr_scheduler_config['gamma']
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.lr_scheduler_config['type'] == "TrueCosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.lr_scheduler_config['T_max'],
                eta_min=self.lr_scheduler_config['eta_min']
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            raise NotImplementedError(f"Scheduler {self.lr_scheduler_config['type']} is not implemented yet.")

    @torch.no_grad()
    def inference(self, lr_tensor: torch.Tensor) -> torch.Tensor:
        net_to_use = self._get_eval_model()
        self.sde.set_model(net_to_use)
        
        h, w = lr_tensor.shape[-2], lr_tensor.shape[-1]
        target_shape = (h * self.scale_factor, w * self.scale_factor)
        mu = self._upscale_lr(lr_tensor, target_shape)
        self.sde.set_mu(mu)
        xt = self.sde.noise_state(mu)
        preds = self.sde.reverse_sde(xt)

        return torch.clamp(preds, 0.0, 1.0)

    @torch.no_grad()
    def log_images(self, batch: dict, N: int = 4, **kwargs) -> dict:
        log = dict()
        
        lr = batch['img'][:N]
        hr = batch['gt'][:N]
        actual_N = lr.shape[0]

        img_names = batch.get('name', [f"image_{i}" for i in range(actual_N)])
        img_names = img_names[:N]

        # Upscale LR to match HR dimensions for SDE processing
        mu = self._upscale_lr(lr, hr.shape)

        net_to_use = self._get_eval_model()
        self.sde.set_model(net_to_use)
        self.sde.set_mu(mu)
        
        xt = self.sde.noise_state(mu)
        preds = self.sde.reverse_sde(xt)

        mu = torch.clamp(mu, 0.0, 1.0)
        preds = torch.clamp(preds, 0.0, 1.0)
        hr = torch.clamp(hr, 0.0, 1.0)

        for i, name in enumerate(img_names):
            # Display: [Upscaled LR (mu), Prediction, HR]
            triplet = torch.stack([mu[i], preds[i], hr[i]], dim=0)
            log[name] = triplet

        return log

    @torch.no_grad()
    def reverse_sde_with_intermediates(self, xt: torch.Tensor, num_states: int = 5, **kwargs) -> tuple:
        """
        Run the reverse SDE process and collect a specific number of intermediate 
        denoising states for visualization purposes.
        """
        # Fix scope: T and other SDE parameters belong to self.sde
        T = self.sde.T 
        x = xt.clone()
        intermediate_states = []
        
        # Calculate the interval to sample intermediate frames evenly
        interval = max(1, T // num_states)

        # Run the reverse diffusion loop
        for t in tqdm(reversed(range(1, T + 1)), desc="Denoising process"):
            # Fix scope: score_fn and reverse_sde_step belong to self.sde
            score = self.sde.score_fn(x, t, **kwargs)
            x = self.sde.reverse_sde_step(x, score, t)

            # Sample intermediate states based on the interval
            if t % interval == 0 or t == 1:
                # Clamp the state to [0, 1] for valid image representation
                clamped_x = torch.clamp(x.clone().detach(), 0.0, 1.0)
                intermediate_states.append(clamped_x)

        # Return final prediction and the collected intermediate states
        return x, intermediate_states