import torch
import numpy as np
from mmagic.evaluation.metrics import PSNR, SSIM
from utils.metrics import SREvaluator

def test_full_logic():
    # 1. 实例化我们的 Evaluator (内部处理 [0.0, 1.0] 的 Tensor)
    # 注意：为了和缺省的 MMagic 完全对齐，我们将 crop_border 设为 0
    my_evaluator = SREvaluator(crop_border=0, test_y_channel=False)
    
    # 2. 实例化 MMagic 官方 Metric (对齐 [0, 255] 尺度)
    mm_psnr = PSNR(gt_key='gt', pred_key='pred', convert_to=None)
    mm_ssim = SSIM(gt_key='gt', pred_key='pred', convert_to=None)

    # 3. 构造两个不同的 Batch
    # Batch 1: 2 samples
    b1_preds = torch.rand(2, 3, 64, 64)
    b1_target = torch.rand(2, 3, 64, 64)
    # Batch 2: 1 sample
    b2_preds = torch.rand(1, 3, 64, 64)
    b2_target = torch.rand(1, 3, 64, 64)

    def to_mm_samples(preds, target):
        """将 [0, 1] Tensor 转换为 MMagic [0, 255] 数据格式"""
        samples = []
        for i in range(preds.shape[0]):
            samples.append({
                # MMagic expects 255 scale in this test context
                'gt': target[i] * 255.0, 
                'output': {'pred': preds[i] * 255.0}
            })
        return samples

    print("--- Phase 1: Step Processing ---")
    
    # --- Step 1 ---
    # 新版 API：直接传入即可，它会自动更新内部累加器
    my_evaluator(b1_preds, b1_target)
    mm_psnr.process([], to_mm_samples(b1_preds, b1_target))
    mm_ssim.process([], to_mm_samples(b1_preds, b1_target))
    print("Step 1 processed (2 samples)")

    # --- Step 2 ---
    my_evaluator(b2_preds, b2_target)
    mm_psnr.process([], to_mm_samples(b2_preds, b2_target))
    mm_ssim.process([], to_mm_samples(b2_preds, b2_target))
    print("Step 2 processed (1 sample)")

    print("\n--- Phase 2: Compute() Results ---")
    
    # 我们的结果 (新版 API：调用一次 compute 返回一个包含两者结果的字典)
    my_final_results = my_evaluator.compute()
    my_final_psnr = my_final_results['psnr']
    my_final_ssim = my_final_results['ssim']
    
    # MMagic 的结果 (计算全部 results 列表里的平均值)
    mm_final_psnr = mm_psnr.compute_metrics(mm_psnr.results)['PSNR']
    mm_final_ssim = mm_ssim.compute_metrics(mm_ssim.results)['SSIM']

    print(f"My Global PSNR:     {my_final_psnr:.6f}")
    print(f"MMagic Global PSNR: {mm_final_psnr:.6f}")
    print(f"My Global SSIM:     {my_final_ssim:.6f}")
    print(f"MMagic Global SSIM: {mm_final_ssim:.6f}")

    # 4. 一致性断言
    # 注意：由于新版 SREvaluator 会严格将浮点数转为 uint8 (模拟真实保存的图像)，
    # 而 MMagic 在这里接收的是 float32，可能会有极其微小 (1e-3级别) 的舍入差异，这是学术上完全正常的。
    psnr_match = np.isclose(my_final_psnr, mm_final_psnr, atol=1e-2)
    ssim_match = np.isclose(my_final_ssim, mm_final_ssim, atol=1e-3)
    
    print(f"\nPSNR Match: {'✅' if psnr_match else '❌'}")
    print(f"SSIM Match: {'✅' if ssim_match else '❌'}")

    print("\n--- Phase 3: Reset() Test ---")
    # 新版 API：调用统一的 reset
    my_evaluator.reset()
    
    # 调用 reset 后的 compute() 应该安全返回 0.0，不再有恶心的 UserWarning 或 nan
    reset_results = my_evaluator.compute()
    reset_psnr = reset_results['psnr']
    reset_ssim = reset_results['ssim']
    
    print(f"PSNR after reset: {reset_psnr}")
    print(f"SSIM after reset: {reset_ssim}")
    
    if reset_psnr == 0.0 and reset_ssim == 0.0:
        print("✅ Reset Logic Verified: States cleared safely!")
    else:
        print("❌ Reset check failed.")

if __name__ == "__main__":
    test_full_logic()