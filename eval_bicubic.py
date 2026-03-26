import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from utils.metrics import SREvaluator

def main():
    # 1. 配置路径 (根据你之前的 YAML 自动填入)
    lr_dir = "/data/RiceSR2024/SAR/dataset/test/LR_bicubic/X4"
    gt_dir = "/data/RiceSR2024/SAR/dataset/test/GT"
    scale = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Bicubic Evaluation on {device}...")

    # 2. 实例化评测器 
    # 极其关键：必须保持 crop_border=scale 和 test_y_channel=True，才能和你的模型做绝对公平的对比！
    evaluator = SREvaluator(crop_border=scale, test_y_channel=True)
    evaluator.to(device)
    
    # 获取所有测试图片名称
    img_names = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.tif'))])
    print(f"Found {len(img_names)} images in test set.")
    
    for img_name in tqdm(img_names, desc="Evaluating Bicubic"):
        gt_path = os.path.join(gt_dir, img_name)
        lr_path = os.path.join(lr_dir, img_name)
        
        # 确保 LR 图片存在
        if not os.path.exists(lr_path):
            print(f"Warning: LR image not found for {img_name}, skipping...")
            continue
            
        # 3. 读取 GT 并归一化到 [0, 1]
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # 4. 读取 LR 并归一化到 [0, 1]
        lr_img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        lr_tensor = torch.from_numpy(lr_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        # 5. 执行 Bicubic 上采样
        # align_corners=False 是超分领域的标准设定
        lr_up = F.interpolate(lr_tensor, scale_factor=scale, mode='bicubic', align_corners=False)
        
        # 防溢出截断
        lr_up = torch.clamp(lr_up, 0.0, 1.0)
        
        # 6. 送入评测器计算单张指标并累加
        gt_tensor = gt_tensor.to(device)
        lr_up = lr_up.to(device)
        evaluator(lr_up, gt_tensor)
        
    # 7. 汇总全局平均成绩
    results = evaluator.compute()
    
    print("\n" + "🔥" * 20)
    print("  Bicubic Baseline Results")
    print("🔥" * 20)
    print(f"Dataset : {gt_dir}")
    print(f"Scale   : X{scale}")
    print(f"PSNR    : {results['psnr']:.4f} dB")
    print(f"SSIM    : {results['ssim']:.4f}")
    print("🔥" * 20)

if __name__ == "__main__":
    main()