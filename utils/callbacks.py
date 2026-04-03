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
        # Removed self._loaded_ema_state as it's no longer needed

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """
        Create the shadow model (EMA) BEFORE the checkpoint is loaded.
        Because 'setup' runs before checkpoint restoration, pl_module.net_ema 
        will be registered as a submodule in time. PyTorch Lightning will then 
        automatically load its weights from the checkpoint's state_dict without 
        throwing 'Unexpected key' errors.
        """
        if self.decay <= 0:
            return
        
        # Check if it already exists to avoid overwriting during multi-stage setups
        if not hasattr(pl_module, 'net_ema'):
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
    