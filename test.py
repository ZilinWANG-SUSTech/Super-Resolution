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


def parse_args():
    parser = argparse.ArgumentParser(description="Testing Pipeline for RGB-Guided SAR SR")
    parser.add_argument(
        "-c", "--config", type=str, default="configs/default_train.yaml",
        help="Path to the YAML configuration file."
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

    # load ckpt
    print(f"Loading checkpoint from {args.ckpt}...")
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("net_ema."):
            # 将 net_ema 的权重名替换为 net，使其直接加载到主网络上
            new_state_dict[k.replace("net_ema.", "net.")] = v
        elif k.startswith("net."):
            # 丢弃原本未经 EMA 平滑的主网络权重
            continue
        else:
            # 保留其他组件的权重 (如损失函数、评价指标等)
            new_state_dict[k] = v
    module.load_state_dict(new_state_dict, strict=False)
    print("Successfully loaded EMA weights into the main network for testing.")


    # 5. Setup Trainer for Testing
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        logger=False, # No need to log to TensorBoard for a simple test run
    )

    # 6. Run Evaluation
    print("=== Starting Testing Evaluation ===")
    trainer.test(module, datamodule=datamodule)
    print("=== Testing Completed ===")

if __name__ == "__main__":
    main()