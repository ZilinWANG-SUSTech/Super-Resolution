import os
import cv2
import argparse
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import models
import engines
from utils import build_engine_cls, EMACallback


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Universal Inference Script")
    parser.add_argument("-c", "--config", type=str, default="configs/default_train.yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Trained checkpoint path.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder containing LR images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Folder to save HR images.")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载配置与模型
    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 正在加载模型至 {device}...")

    image_range = config.data.get('image_range', "0~1") # 0~1, -1~1
 
    lightning_cfg = OmegaConf.to_container(config.lightning_module, resolve=True)
    engine_cls = build_engine_cls(lightning_cfg)
    module = engine_cls(**lightning_cfg['params'])
    # module = engine_cls.load_from_checkpoint(args.ckpt, **lightning_cfg['params'])

    ema_callback = EMACallback(decay=config.get('ema_decay', 0))    

    module.eval()
    module.to(device)

    # 2. 读取待处理图片
    valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_names = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_exts)]
    print(f"📦 找到 {len(image_names)} 张图片，开始统一推理流水线...")
    
    for img_name in tqdm(image_names, desc="Processing"):
        img_path = os.path.join(args.input_dir, img_name)
        
        # --- A. 通用前处理 ---
        lr_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if lr_img is None: continue
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        
        # numpy HWC -> tensor CHW, 归一化到 [0, 1]
        if image_range == "0~1":
            lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_img)).permute(2, 0, 1).float() / 255.0
        elif image_range == "-1~1":
            lr_tensor = torch.from_numpy(np.ascontiguousarray(lr_img)).permute(2, 0, 1).float() / 127.5 - 1.0
        else:
            raise ValueError
        lr_tensor = lr_tensor.unsqueeze(0).to(device) 

        # --- B. 🌟 核心黑盒调用 (统一接口) 🌟 ---
        # 脚本根本不需要知道里面是 Diffusion 还是 ResNet
        hr_tensor = module.inference(lr_tensor)

        # --- C. 通用后处理 ---
        # 确保输出被截断在 [0, 1] 之间，防止溢出产生噪点
        hr_tensor = torch.clamp(hr_tensor, 0.0, 1.0)
        hr_numpy = (hr_tensor.squeeze(0).cpu().numpy() * 255.0).round().astype(np.uint8)        
        
        # tensor CHW -> numpy HWC, RGB -> BGR
        hr_numpy = hr_numpy.transpose(1, 2, 0)
        hr_numpy = cv2.cvtColor(hr_numpy, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(os.path.join(args.output_dir, img_name), hr_numpy)

    print(f"✅ 所有图像已成功处理并保存至 {args.output_dir}")

if __name__ == "__main__":
    main()