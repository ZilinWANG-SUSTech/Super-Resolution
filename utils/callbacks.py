import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import copy

class EMACallback(Callback):
    """
    Exponential Moving Average (EMA) callback tailored for SRRegressionModule.
    """
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Create a shadow model (EMA) for the main model before training starts.
        """
        # Note: Assuming the main network is named 'self.net', we create 'self.net_ema'.
        if self.decay <= 0:
            return
        
        pl_module.net_ema = copy.deepcopy(pl_module.net)
        
        # Always keep the EMA model in evaluation mode to freeze batchnorm/dropout.
        pl_module.net_ema.eval() 
        
        # Freeze gradients for the EMA model as it is updated via moving average.
        for param in pl_module.net_ema.parameters():
            param.requires_grad = False

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        """
        Update the shadow model weights inplace after every optimizer.step().
        """
        if self.decay <= 0:
            return
        
        net_params = dict(pl_module.net.named_parameters())
        net_ema_params = dict(pl_module.net_ema.named_parameters())
        
        for k in net_ema_params.keys():
            net_ema_params[k].data.mul_(self.decay).add_(
                net_params[k].data, alpha=1 - self.decay
            )

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict):
        """
        Inject the EMA weights into the checkpoint dictionary when saving.
        This ensures compatibility with the official DiffIR weight format.
        """
        if self.decay > 0 and hasattr(pl_module, 'net_ema'):
            checkpoint['params_ema'] = pl_module.net_ema.state_dict()

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict):
        """
        Restore the EMA model weights when resuming training or loading pretrained weights.
        """
        if self.decay > 0 and 'params_ema' in checkpoint and hasattr(pl_module, 'net_ema'):
            pl_module.net_ema.load_state_dict(checkpoint['params_ema'])