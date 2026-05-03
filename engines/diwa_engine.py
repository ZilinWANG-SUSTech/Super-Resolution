import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.metrics import SREvaluatorPyIQA
from utils import ENGINE_REGISTRY, build_network
from torch.nn import init
import functools
import copy


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))



@ENGINE_REGISTRY.register()
class DiwaDiffusionEngine(pl.LightningModule):
    def __init__(
        self, 
        network_config: dict,
        diffusion_config: dict,
        optimizer_config: dict,
        beta_schedule: dict,
        eval_crop_border: int = 4, 
        lr_scheduler_config: dict = None,
        img_size: int = None,
    ):
        super().__init__()
        # self.save_hyperparameters()
        self.save_hyperparameters(ignore=['diffusion_config', 'network_config'])
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.beta_schedule = beta_schedule
        self.diffusion_config = diffusion_config
        self.network_config = network_config
        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=True)
        
        diff_cfg = diffusion_config.copy()
        diff_cfg['params']["denoise_fn"] = build_network(network_config)
        self.net = build_network(diff_cfg)
        init_weights(self.net, init_type="orthogonal")
        self.net.set_loss(self.device)
        self.schedule_phase = None
        # self._set_noise_schedule("train")
        # self.net_ema = copy.deepcopy(self.net)
        # self.net_ema.eval()
        # for param in self.net_ema.parameters():
        #         param.requires_grad = False
        # self.net_ema.set_new_noise_schedule(self.beta_schedule['val'], self.device)

    def _set_noise_schedule(self, phase: str):
        schedules = self.beta_schedule
        schedule_opt = schedules.get(phase)
        
        if schedule_opt and self.schedule_phase != phase:
            self.schedule_phase = phase
            model = self.net_ema if hasattr(self, "net_ema") and phase != 'train' else self.net
            model.set_new_noise_schedule(schedule_opt, self.device)
            print(f"Notice: Switched Diffusion Noise Schedule to [{phase}] phase.")

    def on_train_epoch_start(self):
        self._set_noise_schedule('train')

    def on_validation_start(self):
        self._set_noise_schedule('val')
        
    def on_test_start(self):
        self._set_noise_schedule('val')

    def forward(self, x_in: dict) -> torch.Tensor:
        pass

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert tensors from [-1, 1] to [0, 1] range for evaluation and logging.
        """
        return (x + 1.0) / 2.0
    
    def _upsample_lr(self, lr: torch.Tensor, hr_shape: torch.Size) -> torch.Tensor:
        if lr.shape[-2:] != hr_shape[-2:]:
            return F.interpolate(lr, size=hr_shape[-2:], mode='bicubic', align_corners=False)
        return lr

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        hr = batch['gt']
        batch['img'] = self._upsample_lr(batch['img'], hr.shape)
        loss = self.net(batch)
        b, c, h, w = hr.shape
        loss = loss.sum() / int(b * c * h * w)
        
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def _shared_eval_step(self, batch: dict, batch_idx: int, stage: str):
        hr = batch['gt']
        lr = self._upsample_lr(batch['img'], hr.shape)
                
        with torch.no_grad():
            model = self.net_ema if hasattr(self, "net_ema") else self.net
            
            preds = model.super_resolution(lr, continous=False)
            preds_eval = torch.clamp(self._denormalize(preds).detach().float(), 0.0, 1.0)
            hr_eval = torch.clamp(self._denormalize(hr).detach().float(), 0.0, 1.0)
            
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
        if save_dir is not None:
            save_filename = os.path.join(save_dir, "test_results_summary.xlsx")
            self.evaluator.save_to_excel(save_filename, metrics=metrics)
        self.evaluator.reset()

    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        Includes parameter filtering logic for finetune_norm.
        """
        optim_params = []
        
        # Check if finetune_norm flag is active
        finetune_norm = self.network_config.get('finetune_norm', False)
        
        if finetune_norm:
            for k, v in self.net.named_parameters():
                v.requires_grad = False
                # Filter specific parameters for optimization
                if k.find('transformer') >= 0:
                    v.requires_grad = True
                    v.data.zero_()  # Reset initial weights to 0
                    optim_params.append(v)
            print("Notice: [finetune_norm] is True. Only transformer params will be optimized.")
        else:
            # Optimize all parameters
            optim_params = list(self.net.parameters())

        # Optimizer settings
        opt_type = self.optimizer_config.get("type", "adam").lower()
        lr = self.optimizer_config.get("lr", 1e-4)
        
        if opt_type == "adamw":
            weight_decay = self.optimizer_config.get("weight_decay", 0.01)
            optimizer = torch.optim.AdamW(optim_params, lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(optim_params, lr=lr)

        # Scheduler settings
        if self.lr_scheduler_config is None:
            return optimizer
            
        if self.lr_scheduler_config.get('type') == "MultiStepLR":
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
    def inference(self, lr_tensor: torch.Tensor) -> torch.Tensor:
        model = self.net_ema if hasattr(self, "net_ema") else self.net
        
        # Pad spatial dimensions to be multiples of 16
        _, _, h_old, w_old = lr_tensor.size()
        pad_size = 16
        h_pad = (pad_size - h_old % pad_size) % pad_size
        w_pad = (pad_size - w_old % pad_size) % pad_size
        
        if h_pad or w_pad:
            # Reflection padding
            lr_tensor_padded = F.pad(lr_tensor, (0, w_pad, 0, h_pad), mode='reflect')
            
            # Reverse diffusion inference
            preds = model.super_resolution(lr_tensor_padded, continous=False)   

            scale = getattr(self.hparams, 'eval_crop_border', 4)
            h_hr_true = h_old * scale
            w_hr_true = w_old * scale

            # Crop the padded areas back to the original target shape
            preds = preds[..., :h_hr_true, :w_hr_true]
        else:
            preds = model.super_resolution(lr_tensor, continous=False)

        # Denormalize [-1, 1] -> [0, 1] and clamp
        return torch.clamp(self._denormalize(preds), 0.0, 1.0)

    @torch.no_grad()
    def log_images(self, batch: dict, N: int = 4, **kwargs) -> dict:
        """
        Log images for TensorBoard/Wandb visualization.
        Returns a dictionary of triplets: [LR_Upsampled, SR_Diffusion_Prediction, HR_Ground_Truth].
        Values are mapped to [0, 1].
        """
        log = dict()

        hr = batch['gt'][:N]
        lr = batch['img'][:N]
        lr = self._upsample_lr(lr, hr.shape)
        actual_N = lr.shape[0]

        img_names = batch.get('name', [f"image_{i}" for i in range(actual_N)])
        img_names = img_names[:N]

        net_to_use = self.net_ema if hasattr(self, "net_ema") else self.net
        
        # Reverse diffusion inference
        preds = net_to_use.super_resolution(lr, continous=False)

        # Upsample LR to target resolution using bicubic interpolation for visual alignment
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False)

        # Denormalize [-1, 1] -> [0, 1] and clamp all tensors before logging
        lr_up = torch.clamp(self._denormalize(lr_up), 0.0, 1.0)
        preds = torch.clamp(self._denormalize(preds), 0.0, 1.0)
        hr = torch.clamp(self._denormalize(hr), 0.0, 1.0)

        for i, name in enumerate(img_names):
            triplet = torch.stack([lr_up[i], preds[i], hr[i]], dim=0)
            log[name] = triplet

        return log
    