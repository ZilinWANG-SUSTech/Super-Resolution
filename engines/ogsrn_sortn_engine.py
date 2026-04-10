import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.metrics import SREvaluatorPyIQA
from utils import ENGINE_REGISTRY, build_network


@ENGINE_REGISTRY.register()
class SORTNModule(pl.LightningModule):
    def __init__(
        self,
        network_g_config: dict,
        network_d_config: dict,
        optimizer_g_config: dict,
        optimizer_d_config: dict,
        alpha: float = 100.0,
        eval_crop_border: int = 0, 
        lr_scheduler_g_config: dict = None,
        lr_scheduler_d_config: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.automatic_optimization = False

        self.optimizer_g_config = optimizer_g_config
        self.optimizer_d_config = optimizer_d_config
        self.lr_scheduler_g_config = lr_scheduler_g_config
        self.lr_scheduler_d_config = lr_scheduler_d_config
        self.alpha = alpha

        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=False)
        
        self.net_g = build_network(network_g_config)
        self.net_d = build_network(network_d_config)
        
        self.cri_gan = nn.BCEWithLogitsLoss()

    def forward(self, sar_img: torch.Tensor) -> torch.Tensor:
        return self.net_g(sar_img)

    def training_step(self, batch: dict, batch_idx: int) -> None:
        optic_real = batch['guide']
        sar_input = batch['hr']
        batch_size = optic_real.size(0)
        
        opt_g, opt_d = self.optimizers()

        label_real = torch.ones((batch_size,), dtype=torch.float, device=self.device)
        label_fake = torch.zeros((batch_size,), dtype=torch.float, device=self.device)

        # =========================================================
        # 1. Train Generator (SORTN)
        # target: maximize log(D(G(x))) - alpha * L1(y, G(x))
        # =========================================================
        self.toggle_optimizer(opt_g)
        
        optic_gen, _ = self.net_g(sar_input)
            
        log_dict_g = {}
        
        
        l1_loss = F.l1_loss(optic_gen, optic_real)
        
        pred_fake, _ = self.net_d(optic_gen)
        gan_loss = self.cri_gan(pred_fake.flatten(), label_real) 
        
        # Generator All Loss
        errG = gan_loss + self.alpha * l1_loss
        
        log_dict_g["train/g_l1_loss"] = l1_loss
        log_dict_g["train/g_gan_loss"] = gan_loss
        log_dict_g["train/g_total"] = errG
        
        self.manual_backward(errG)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # =========================================================
        # 2. Train discriminator (PatchGAN)
        # Target: maximize log(D(y)) + log(1 - D(G(x)))
        # =========================================================
        self.toggle_optimizer(opt_d)
        
        log_dict_d = {}

        # True Image Loss：Output 1
        pred_real, _ = self.net_d(optic_real)
        errD_real = self.cri_gan(pred_real.flatten(), label_real)
        
        # Fake Image Loss：Output 0
        pred_fake_detached, _ = self.net_d(optic_gen.detach())
        errD_fake = self.cri_gan(pred_fake_detached.flatten(), label_fake)
        
        # Discriminator Loss
        errD = errD_real + errD_fake
        
        log_dict_d["train/d_real"] = errD_real
        log_dict_d["train/d_fake"] = errD_fake
        log_dict_d["train/d_total"] = errD

        self.manual_backward(errD)
        opt_d.step()
        opt_d.zero_grad()
            
        self.untoggle_optimizer(opt_d)

        # =========================================================
        # 3. Metrics
        # =========================================================
        with torch.no_grad():
            preds_eval = torch.clamp(optic_gen.detach().float(), 0.0, 1.0)
            hr_eval = torch.clamp(optic_real.detach().float(), 0.0, 1.0)
            train_psnr = self.evaluator.psnr(preds_eval, hr_eval).mean().item()
            
        full_log = {**log_dict_g, **log_dict_d, "train/psnr_epoch": train_psnr}
        self.log_dict(full_log, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)

    def on_train_epoch_end(self):
        sch_g, sch_d = self.lr_schedulers()
        sch_g.step()
        sch_d.step()

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        optic_real = batch['guide']
        sar_input = batch['hr']
                
        with torch.no_grad():
            output, _ = self.net_g(sar_input)
            preds = output.sample if hasattr(output, 'sample') else output
            
            preds_eval = torch.clamp(preds.float(), 0.0, 1.0)
            hr_eval = torch.clamp(optic_real.float(), 0.0, 1.0)

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
        if save_dir:
            save_filename = os.path.join(save_dir, "test_results_summary.xlsx")
            self.evaluator.save_to_excel(save_filename, metrics=metrics)
        self.evaluator.reset()

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.net_g.parameters(), **self.optimizer_g_config)
        opt_d = torch.optim.Adam(self.net_d.parameters(), **self.optimizer_d_config)
        
        if self.lr_scheduler_g_config['type'] == "LinearDecay":
            decay_epoch = self.lr_scheduler_g_config.get('decay_after_epoch', 50)
            
            lr_lambda = lambda epoch: 1.0 if epoch <= decay_epoch else max(0.0, 1.0 - (epoch - decay_epoch) / 100.01)
            sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
            sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)
            
        elif self.lr_scheduler_g_config['type'] == "MultiStepLR":
            sch_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, **self.lr_scheduler_g_config['kwargs'])
            sch_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, **self.lr_scheduler_d_config['kwargs'])
        else:
            raise NotImplementedError(f"Scheduler type not implemented.")

        return [opt_g, opt_d], [sch_g, sch_d]