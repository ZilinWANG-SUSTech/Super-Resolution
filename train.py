import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from data.datamodule import SRDataModule
import models
import engines
from utils import EMACallback, build_engine, ImageLogger, SRImageLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Training Pipeline for RGB-Guided SAR SR")
    
    # Core configurations
    parser.add_argument(
        "-c", "--config", type=str, default="configs/default_train.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "-n", "--name", type=str, default="default_exp",
        help="Name of the experiment. Used for logging and saving checkpoints."
    )
    
    # Training controls
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint (.ckpt) to resume training from."
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=None,
        help="Override the GPU indices to use, e.g., --gpus 0 1"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run a fast development run (1 train/val step) to check for bugs."
    )
    
    return parser.parse_args()

def main():
    # 1. Parse CLI arguments
    args = parse_args()

    # 2. Load YAML Configuration using OmegaConf
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found!")
    config = OmegaConf.load(args.config)

    # 3. Setup Seed for reproducibility
    pl.seed_everything(42, workers=True)

    # 4. Instantiate DataModule
    # Convert OmegaConf DictConfig to native python dict for **kwargs unpacking
    data_cfg = OmegaConf.to_container(config.data, resolve=True)
    datamodule = SRDataModule(**data_cfg)

    # 5. Instantiate LightningModule
    lightning_cfg = OmegaConf.to_container(config.lightning_module, resolve=True)
    module = build_engine(lightning_cfg)


    # 6. Setup Logger
    logger = TensorBoardLogger(save_dir="logs", name=args.name)
    
    # 7. Setup Callbacks
    # Save the best model based on the validation PSNR

    trainer_cfg = OmegaConf.to_container(config.trainer, resolve=True)
    
    # Override trainer config with CLI arguments if provided
    if args.gpus is not None:
        trainer_cfg['devices'] = args.gpus
    if args.debug:
        trainer_cfg['fast_dev_run'] = True

    # TODO: SR
    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="best-{epoch:03d}-{val/psnr:.4f}",
        monitor="val/psnr",
        mode="max",
        save_top_k=3,           # Keep the top 3 best models
        save_last=True,         # Always save the latest model to easily resume
    )

    # # TODO: Autoencoder
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=os.path.join("checkpoints", args.name),
    #     filename="{epoch:06}-{step:09}",
    #     verbose=True,
    #     save_top_k=-1,
    #     every_n_train_steps=10000,
    #     save_weights_only=True
    # )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # TODO: SR
    early_stop_callback = EarlyStopping(
        monitor="val/psnr", 
        min_delta=0, 
        patience=40,  # 20 Val = 60 Epoch
        mode="max", 
        verbose=True
    )
    
    ema_callback = EMACallback(decay=config.get('ema_decay', 0))

    image_logger_cfg = config.get("image_logger", {})
    if image_logger_cfg:
        img_logger = SRImageLogger(
            **image_logger_cfg
        )
    else:
        img_logger = None

    callbacks = [checkpoint_callback, lr_monitor, early_stop_callback, ema_callback]
    # callbacks = [checkpoint_callback, lr_monitor, ema_callback]
    if img_logger:
        callbacks.append(img_logger)

    # 8. Setup Trainer
    trainer = pl.Trainer(
        **trainer_cfg,
        logger=logger,
        callbacks=callbacks,
    )

    # 9. Start Training (and optionally Testing)
    print(f"=== Starting Training for Experiment: {args.name} ===")
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.resume)

    # Optionally, automatically run the test set after training finishes using the best checkpoint
    if hasattr(datamodule, 'test_dataset') and datamodule.test_dataset is not None:
        print("=== Starting Testing on the Best Model ===")
        trainer.test(module, datamodule=datamodule, ckpt_path="best")

if __name__ == "__main__":
    main()