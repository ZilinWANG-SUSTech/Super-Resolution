import os
import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf

from data.datamodule import SRDataModule
from engines.diffusion_engine import SRDiffusionModule
from engines.regression_engine import SRRegressionModule
import models
import engines
from utils import EMACallback, build_engine_cls
import torch
from pytorch_lightning.loggers import TensorBoardLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Testing Pipeline for RGB-Guided SAR SR")
    parser.add_argument(
        "-c", "--config", type=str, default="configs/default_train.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "-n", "--name", type=str, default="default_exp",
        help="Name of the experiment. Used for logging and saving checkpoints."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to the trained checkpoint (.ckpt file)."
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0],
        help="GPU indices to use for testing."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = OmegaConf.load(args.config)

    # 2. Setup Seed
    pl.seed_everything(42, workers=True)

    # 3. Instantiate DataModule
    # We only need the test dataset here, but DataModule handles everything
    data_cfg = OmegaConf.to_container(config.data, resolve=True)
    datamodule = SRDataModule(**data_cfg)

    # 4. Load Model from Checkpoint
    # PTL magically restores all hyperparameters (network_config, etc.) 
    # saved during training via self.save_hyperparameters()
    lightning_cfg = OmegaConf.to_container(config.lightning_module, resolve=True)
    module = build_engine_cls(lightning_cfg)
    module = module(**lightning_cfg.get('params', {}))
    # module = module.load_from_checkpoint(args.ckpt, **lightning_cfg['params'])
    
    # 5. Setup Logger & Callbacks
    logger = TensorBoardLogger(save_dir="logs", name=args.name)
    ema_callback = EMACallback(decay=config.get('ema_decay', 0))

    # 6. Setup Trainer for Testing
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        logger=logger,
        callbacks=[ema_callback]
    )

    # 6. Run Evaluation
    print("=== Starting Testing Evaluation ===")
    trainer.test(module, datamodule=datamodule, ckpt_path=args.ckpt)
    print("=== Testing Completed ===")

if __name__ == "__main__":
    main()