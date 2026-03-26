import torch

def inspect_checkpoint(ckpt_path):
    print(f"正在解剖 Checkpoint: {ckpt_path}\n" + "="*50)
    
    # 1. 将模型加载到 CPU (防止爆显存)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # 2. 检查最外层的骨架
    print("\n[第一层：顶层 Keys (PL 原生框架信息)]")
    for key in ckpt.keys():
        if key not in ['state_dict', 'params_ema']:
            # 打印非权重信息，比如 epoch, global_step, optimizer_states 等
            print(f" - {key}")
            
    # 3. 验证你的 Callback 是否发挥了魔法
    print("\n[第二层：验证 EMACallback 显式注入]")
    if 'params_ema' in ckpt:
        print(f" ✅ 成功找到 'params_ema'！(共包含 {len(ckpt['params_ema'])} 个参数层)")
    else:
        print(" ❌ 警告：未找到 'params_ema'！请检查 Callback 是否被正确挂载到了 Trainer。")

    # 4. 潜入 state_dict 检查 PL 的自动抓取机制
    print("\n[第三层：解剖 state_dict 内部的网格]")
    if 'state_dict' in ckpt:
        state_dict_keys = list(ckpt['state_dict'].keys())
        
        # 统计以 net. 和 net_ema. 开头的权重数量
        net_keys = [k for k in state_dict_keys if k.startswith('net.')]
        ema_keys = [k for k in state_dict_keys if k.startswith('net_ema.')]
        other_keys = [k for k in state_dict_keys if not (k.startswith('net.') or k.startswith('net_ema.'))]
        
        print(f" - 发现 'net.' (主网络) 参数: {len(net_keys)} 个")
        print(f" - 发现 'net_ema.' (影子网络) 参数: {len(ema_keys)} 个")
        if other_keys:
            print(f" - 发现其他未知参数: {len(other_keys)} 个 (例如: {other_keys[:3]}...)")
            
        if len(net_keys) > 0 and len(net_keys) == len(ema_keys):
            print(" ✅ 完美！主网络与影子网络均被 PL 自动抓取，且层数完全对齐！")
    else:
        print(" ❌ 警告：未找到 'state_dict'！这是一个极其不正常的检查点文件。")

if __name__ == "__main__":
    # 把这里换成你刚刚跑出来的 ckpt 绝对或相对路径
    YOUR_CKPT_PATH = "checkpoints/DiffIR/S1/best-epoch=089-val_psnr=26.16.ckpt" 
    inspect_checkpoint(YOUR_CKPT_PATH)