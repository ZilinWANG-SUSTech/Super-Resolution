import os
import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tqdm import tqdm

# 直接复用你项目里的原生组件
from data.datamodule import SRDataModule
from utils import build_engine

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate scale_factor for Latent Space")
    parser.add_argument(
        "-c", "--config", type=str, default="configs/default_train.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "-b", "--batches", type=int, default=20,
        help="Number of batches to compute the global standard deviation."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} not found!")
    
    print(f"=== Loading Config: {args.config} ===")
    config = OmegaConf.load(args.config)

    # 1. 锁随机种子，保证每次抽样算出来的方差一致
    pl.seed_everything(42, workers=True)

    # 2. 实例化 DataModule (完美对齐 train.py)
    print("=== Initializing DataModule ===")
    data_cfg = OmegaConf.to_container(config.data, resolve=True)
    datamodule = SRDataModule(**data_cfg)
    # Trainer 通常会自动调 setup()，离线脚本需要我们手动调一下
    datamodule.setup(stage='fit') 
    train_loader = datamodule.train_dataloader()

    # 3. 实例化 LightningModule (完美对齐 train.py)
    print("=== Initializing LDM Engine & VAE ===")
    lightning_cfg = OmegaConf.to_container(config.lightning_module, resolve=True)
    model = build_engine(lightning_cfg)
    
    # 抽取 LDM 内部已经加载好权重的 VAE，并推到 GPU
    model = model.eval().cuda()
    vae = model.first_stage_model
    first_stage_key = model.first_stage_key

    # 4. 跑数据流算方差
    print(f"=== Extracting latents for {args.batches} batches ===")
    z_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader, total=args.batches)):
            if i >= args.batches:
                break
                
            # 拿到高清原图送入 GPU
            x = batch[first_stage_key].cuda()
            
            # 编码进入潜空间
            encoder_posterior = vae.encode(x)
            
            # LDM 默认取 mode 作为确定的隐特征
            z = encoder_posterior.mode().detach()
            
            z_list.append(z.flatten())

    # 5. 汇总计算
    print("=== Calculating global standard deviation ===")
    z_all = torch.cat(z_list, dim=0)
    std = z_all.std().item()
    
    # 核心公式
    scale_factor = 1.0 / std

    print("\n" + "="*60)
    print(f"✅ 潜空间全局标准差 (Std): {std:.6f}")
    print(f"🎯 强烈建议的 scale_factor: {scale_factor:.6f}")
    print("="*60 + "\n")
    print(f"【终极替换指南】:")
    print(f"请打开 {args.config}，在 model -> params 下修改：")
    print(f"    scale_factor: {scale_factor:.6f}")
    print(f"    scale_by_std: False")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()