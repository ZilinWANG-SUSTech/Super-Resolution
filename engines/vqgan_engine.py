import torch
import pytorch_lightning as pl
from utils.metrics import SREvaluatorPyIQA
from utils import ENGINE_REGISTRY, build_network
import importlib


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


@ENGINE_REGISTRY.register()
class VQGANModule(pl.LightningModule):
    def __init__(
        self,
        network_config: dict,    # Config for the new VQGANNetwork
        lossconfig: dict,        # Config for VQLPIPSWithDiscriminator
        optimizer_g_config: dict,
        optimizer_d_config: dict,
        image_key: str = "gt",
        eval_crop_border: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Consistent with your gan_engine: manual optimization for GAN stability
        self.automatic_optimization = False

        self.image_key = image_key
        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=True)
        
        # 1. Build the unified VQGAN Network (The one we just encapsulated)
        self.net = build_network(network_config)
        
        # 2. Build Unified Loss (Handles Discriminator and LPIPS internally)
        self.loss = instantiate_from_config(lossconfig)

    def training_step(self, batch: dict, batch_idx: int):
        # Input x is typically in [-1, 1] from your SAR dataloader
        x = batch[self.image_key]
        opt_ae, opt_disc = self.optimizers()

        # ---------------------------------------------------------
        # 1. Optimize Autoencoder (net: Encoder + Decoder + Quantizer)
        # ---------------------------------------------------------
        self.toggle_optimizer(opt_ae)
        
        # Use the streamlined forward pass of the encapsulated network
        x_rec, qloss = self.net(x)
        
        # optimizer_idx=0 signals the loss module to compute Generator-side loss
        aeloss, log_dict_ae = self.loss(
            qloss, x, x_rec, optimizer_idx=0, 
            global_step=self.global_step,
            last_layer=self.net.decoder.conv_out.weight, # Access nested weight
            split="train"
        )

        self.manual_backward(aeloss)
        opt_ae.step()
        opt_ae.zero_grad()
        self.untoggle_optimizer(opt_ae)

        # ---------------------------------------------------------
        # 2. Optimize Discriminator
        # ---------------------------------------------------------
        self.toggle_optimizer(opt_disc)
        
        # optimizer_idx=1 signals the loss module to compute Discriminator-side loss
        discloss, log_dict_disc = self.loss(
            qloss, x, x_rec, optimizer_idx=1, 
            global_step=self.global_step,
            last_layer=self.net.decoder.conv_out.weight, 
            split="train"
        )

        self.manual_backward(discloss)
        opt_disc.step()
        opt_disc.zero_grad()
        self.untoggle_optimizer(opt_disc)

        # Unified logging
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True)
        self.log("train/aeloss", aeloss, prog_bar=True)

    def validation_step(self, batch: dict, batch_idx: int):
        x = batch[self.image_key]
        x_rec, qloss = self.net(x)
        
        # Validation loss calculation
        aeloss, _ = self.loss(
            qloss, x, x_rec, optimizer_idx=0, 
            global_step=self.global_step,
            last_layer=self.net.decoder.conv_out.weight, # Access nested weight
            split="val"
        )
        
        # Metrics: map back to [0, 1] domain
        preds_eval = torch.clamp(x_rec.detach().float() * 0.5 + 0.5, 0.0, 1.0)
        hr_eval = torch.clamp(x.detach().float() * 0.5 + 0.5, 0.0, 1.0)
        self.evaluator(preds_eval, hr_eval)
        
        self.log("val/aeloss", aeloss, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        metrics = self.evaluator.compute()
        for k, v in metrics.items():
            self.log(f"val/{k}", v, prog_bar=True, sync_dist=True)
        self.evaluator.reset()

    def configure_optimizers(self):
        # Now all AE parameters are neatly organized under self.net
        opt_ae = torch.optim.Adam(self.net.parameters(), **self.hparams.optimizer_g_config)
        
        # Discriminator parameters remain inside the loss module
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), 
            **self.hparams.optimizer_d_config
        )
        
        return [opt_ae, opt_disc], []

    @torch.no_grad()
    def log_images(self, batch: dict, N: int = 4, **kwargs) -> dict:
        log = dict()
        x = batch[self.image_key][:N]
        x_rec, _ = self.net(x)
        
        log["inputs"] = torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)
        log["reconstructions"] = torch.clamp(x_rec * 0.5 + 0.5, 0.0, 1.0)
        return log