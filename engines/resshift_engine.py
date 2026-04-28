import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.metrics import SREvaluatorPyIQA
from utils import ENGINE_REGISTRY, build_network, build_diffusion
import os
import numpy as np
import torchvision.utils as vutils
from einops import rearrange


@ENGINE_REGISTRY.register()
class ResShiftDiffusionModule(pl.LightningModule):
    def __init__(
        self, 
        network_config: dict,
        optimizer_config: dict,
        diffusion_config: dict,       
        autoencoder_config: dict = None,
        autoencoder_ckpt_path: str = None,
        eval_crop_border: int = 4, 
        lr_scheduler_config: dict = None,
        val_y_channel: bool = True
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_config = optimizer_config
        self.lr_scheduler_config = lr_scheduler_config
        self.evaluator = SREvaluatorPyIQA(crop_border=eval_crop_border, test_y_channel=val_y_channel)
        
        # 1. Build the main network
        self.net = build_network(network_config)

        # 2. Build the Autoencoder using build_network
        if autoencoder_config is not None:
            self.autoencoder = build_network(autoencoder_config)
            
            # Load pre-trained weights for the autoencoder using the newly provided argument
            if autoencoder_ckpt_path is not None:
                if os.path.exists(autoencoder_ckpt_path):
                    ae_ckpt = self.load_ae_ckpt(autoencoder_ckpt_path)
                    self.autoencoder.load_state_dict(ae_ckpt, strict=True)
            else:
                raise FileNotFoundError(f"Autoencoder checkpoint not found at {autoencoder_ckpt_path}")

            # Freeze autoencoder parameters
            for params in self.autoencoder.parameters():
                params.requires_grad_(False)
            self.autoencoder.eval()
        else:
            self.autoencoder = None

        # 3. Build the Diffusion core controller directly
        self.base_diffusion = build_diffusion(diffusion_config)

    def load_ae_ckpt(self, ckpt_path):
        """
        Load and clean the autoencoder checkpoint by removing unneeded modules 
        (like evaluators) and stripping the 'net.' prefix.
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Autoencoder checkpoint not found at {ckpt_path}")

        # Load the raw checkpoint to CPU to avoid VRAM spikes
        raw_ckpt = torch.load(ckpt_path, map_location='cpu')

        # Extract state_dict if wrapped in a training checkpoint format
        if 'state_dict' in raw_ckpt:
            raw_ckpt = raw_ckpt['state_dict']

        clean_state_dict = {}
        
        # Iterate over the original keys to filter and rename
        for key, value in raw_ckpt.items():
            # Only keep the weights belonging to the core autoencoder network
            if key.startswith('net.'):
                # Strip the 'net.' prefix (which is exactly 4 characters long)
                new_key = key[4:]
                clean_state_dict[new_key] = value

        return clean_state_dict

    def configure_optimizers(self):
        print(self.optimizer_config)
        optimizer = torch.optim.AdamW(self.net.parameters(), **self.optimizer_config)
        
        if self.lr_scheduler_config['type'] == "cosin":
            warmup_epochs = self.lr_scheduler_config['warmup_epoch']
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer,
                start_factor=1.0 / warmup_epochs if warmup_epochs > 0 else 1.0, 
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.lr_scheduler_config['T_max'],
                eta_min=self.lr_scheduler_config['eta_min'],
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer=optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs]
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

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # TODO: Image Range: -1~1
        hr = batch['gt']
        lr = batch['img']

        # 1. Sample timesteps tt
        tt = torch.randint(
            0, self.base_diffusion.num_timesteps,
            size=(hr.shape[0],),
            device=self.device,
        )

        # 2. Generate noise with dynamically matched dimensions
        ae_params = self.hparams.autoencoder_config.get('params', {})
        ddconfig = ae_params.get('ddconfig', {})
        ch_mult = ddconfig.get('ch_mult', None)
        latent_downsampling_sf = 2 ** (len(ch_mult) - 1)
        latent_h = hr.shape[2] // latent_downsampling_sf
        latent_w = hr.shape[3] // latent_downsampling_sf
        noise_chn = ae_params['embed_dim']
        noise = torch.randn(
                size=(hr.shape[0], noise_chn, latent_h, latent_w),
                device=self.device,
            )

        # 3. Construct kwargs for the model
        model_kwargs = {'lq': lr}

        # 4. Calculate forward process and get predictions via diffusion wrapper
        losses_dict, z_t, z0_pred = self.base_diffusion.training_losses(
            self.net,          
            hr,                
            lr,                
            tt,                
            first_stage_model=self.autoencoder, 
            model_kwargs=model_kwargs,
            noise=noise,
        )

        # In TrainerDifIR, total loss is strictly losses['mse'] (Ref: trainer.py line 510)
        loss = losses_dict['mse'].mean()

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    @torch.no_grad()
    def inference(self, im_lq: torch.Tensor, save_progress: bool = False):
        model = self.net_ema if hasattr(self, "net_ema") else self.net
        
        scale = self.hparams.eval_crop_border
        _, _, h_old, w_old = im_lq.shape

        window_size = self.hparams.network_config["params"]["window_size"]
        unet_downsample = 8
        latent_target_mod = window_size * unet_downsample
        hr_target_mod = latent_target_mod * scale
        lcm_val = math.lcm(scale, hr_target_mod)
        mod = lcm_val // scale

        pad_h = (mod - h_old % mod) % mod
        pad_w = (mod - w_old % mod) % mod
        if pad_h > 0 or pad_w > 0:
            im_lq = F.pad(im_lq, (0, pad_w, 0, pad_h), mode='reflect')


        # Calculate exactly which timesteps to capture for visualization
        indices = []
        if save_progress:
            num_T = self.base_diffusion.num_timesteps
            indices = np.linspace(
                0, num_T,
                num_T if num_T < 5 else 4,
                endpoint=False,
                dtype=np.int64
            ).tolist()
            if not (num_T - 1) in indices:
                indices.append(num_T - 1)

        progress_images = []
        num_iters = 0
        
        # Progressive denoising loop
        for sample in self.base_diffusion.p_sample_loop_progressive(
            y=im_lq,
            model=model,
            first_stage_model=self.autoencoder,
            noise=None,
            clip_denoised=True if self.autoencoder is None else False,
            model_kwargs={'lq': im_lq},
            device=self.device,
            progress=False,
        ):
            sample_decode = self.base_diffusion.decode_first_stage(
                sample['sample'],
                self.autoencoder,
            ).clamp(-1.0, 1.0)
            
            # Capture intermediate states if required
            if save_progress and num_iters in indices:
                progress_images.append(sample_decode * 0.5 + 0.5)
                
            num_iters += 1
            
        preds = sample_decode * 0.5 + 0.5
        preds = torch.clamp(preds, 0.0, 1.0)
        
        if pad_h > 0 or pad_w > 0:
            hr_h = h_old * scale
            hr_w = w_old * scale

            preds = preds[:, :, :hr_h, :hr_w]
            
            if save_progress:
                progress_images = [img[:, :, :hr_h, :hr_w] for img in progress_images]

        if save_progress:
            return preds, progress_images
        return preds
    
    def validation_step(self, batch: dict, batch_idx: int):
        hr = batch['gt']
        lr = batch['img']
        
        # Only generate and log progressive images for the very first batch 
        # to avoid massive TensorBoard file sizes and save inference time
        log_images_this_batch = (batch_idx == 0)
        
        if log_images_this_batch:
            preds, progress_images = self.inference(lr, save_progress=True)
        else:
            preds = self.inference(lr, save_progress=False)
            
        # Metric evaluation
        preds_eval = torch.clamp(preds.detach().float(), 0.0, 1.0)
        hr_eval = torch.clamp(hr.detach().float() * 0.5 + 0.5, 0.0, 1.0)
        self.evaluator(preds_eval, hr_eval)
        
        # Image Logging to TensorBoard
        if log_images_this_batch and self.global_rank == 0:

            # Format: [Batch, K_steps, Channels, Height, Width]
            progress_stack = torch.stack(progress_images, dim=1) 
            # Flatten to: [(Batch * K_steps), Channels, Height, Width] for the grid
            progress_grid = rearrange(progress_stack, 'b k c h w -> (b k) c h w')
            
            tensorboard = self.logger.experiment
            
            # Keep visuals in [0, 1] domain
            log_lr = torch.clamp(lr * 0.5 + 0.5, 0.0, 1.0)
            log_hr = torch.clamp(hr * 0.5 + 0.5, 0.0, 1.0)
            
            tensorboard.add_image("val/1_LQ", vutils.make_grid(log_lr, nrow=4), self.global_step)
            # The nrow length ensures one progressive sequence per row
            tensorboard.add_image("val/2_Progressive_Denoising", vutils.make_grid(progress_grid, nrow=len(progress_images)), self.global_step)
            tensorboard.add_image("val/3_Prediction", vutils.make_grid(preds_eval, nrow=4), self.global_step)
            tensorboard.add_image("val/4_Ground_Truth", vutils.make_grid(log_hr, nrow=4), self.global_step)


    def on_validation_epoch_end(self):
        # Lightning handles the mean calculation across all batches internally
        metrics = self.evaluator.compute()
        for k, v in metrics.items():
            if k == 'fid': continue
            self.log(f"val/{k}", v, prog_bar=True, sync_dist=True)
        self.evaluator.reset()

    def test_step(self, batch: dict, batch_idx: int):
        hr = batch['gt']
        lr = batch['img']
        with torch.no_grad():
            # In test/eval stage, we only need the final prediction, 
            # so we explicitly set save_progress=False to save memory and time
            preds = self.inference(lr, save_progress=False)
            
            # Ensure tensors are strictly bounded in [0, 1] for metric calculation
            preds_eval = torch.clamp(preds.detach().float(), 0.0, 1.0)
            hr_eval = torch.clamp(hr.detach().float() * 0.5 + 0.5, 0.0, 1.0)
            
        self.evaluator(preds_eval, hr_eval)


    def on_test_epoch_end(self):
        # 1. Compute all accumulated metrics (PSNR, SSIM, LPIPS, etc.)
        metrics = self.evaluator.compute()
        
        # 2. Log metrics to TensorBoard/Wandb
        for k, v in metrics.items():
            self.log(f"test/{k}", v, prog_bar=False, sync_dist=True)
            
        # 3. Save detailed results to Excel (Inherited from your Regression Engine)
        # Add a safety check for the logger directory
        if self.logger and hasattr(self.logger, 'log_dir') and self.logger.log_dir is not None:
            save_dir = self.logger.log_dir
        else:
            save_dir = "./" # Safe fallback
            
        save_filename = os.path.join(save_dir, "test_results_summary.xlsx")
        
        # Important: Ensure only the main process (rank 0) writes the file to disk
        # This prevents file corruption and race conditions in Multi-GPU (DDP) training
        if self.global_rank == 0:
            # Assuming your SREvaluatorPyIQA has the save_to_excel method implemented
            self.evaluator.save_to_excel(save_filename, metrics=metrics)
            print(f"\n[Test Completed] Results successfully exported to: {save_filename}")
            
        # 4. Reset evaluator for future runs
        self.evaluator.reset()


    @torch.no_grad()
    def log_images(self, batch: dict, N: int = 4, **kwargs) -> dict:
        """
        Extracts image sequences (LR, Progressive Denoising steps, Prediction, HR) 
        and groups them by filename for external Callbacks.
        
        Args:
            batch: The validation or training batch.
            N: Maximum number of images to log from the batch.
        """
        log = dict()
        
        lr = batch['img'][:N]
        hr = batch['gt'][:N]
        actual_N = lr.shape[0]

        # 1. Safely extract image filenames
        img_names = batch.get('name', [f"image_{i}" for i in range(actual_N)])
        img_names = img_names[:N]

        # 2. Get model predictions AND progressive denoising steps
        # Utilizing the save_progress flag implemented in inference
        preds, progress_images = self.inference(lr, save_progress=True)

        # 3. Upsample LR image to match HR/SR dimensions for visualization
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False)

        # 4. Clamp strictly to [0, 1] domain
        lr_up = torch.clamp(lr_up * 0.5 + 0.5, 0.0, 1.0)
        preds = torch.clamp(preds, 0.0, 1.0)
        hr = torch.clamp(hr * 0.5 + 0.5, 0.0, 1.0)

        # 5. Group by image name
        for i, name in enumerate(img_names):
            # Extract the i-th image from each progressive step tensor
            prog_seq = [step_tensor[i] for step_tensor in progress_images]
            
            # Create a sequence list: [LR_up] -> [Noise...Clear] -> [Final_Pred] -> [HR]
            sequence = [lr_up[i]] + prog_seq + [preds[i], hr[i]]
            
            # Stack into a single tensor of shape: (Sequence_Length, C, H, W)
            log[name] = torch.stack(sequence, dim=0)

        return log
        