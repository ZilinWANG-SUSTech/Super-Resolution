import os
import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image

from engines import IRSDEEngine 

def load_ema_weights(model: IRSDEEngine, ckpt_path: str, ema_prefix: str = "net_ema.") -> IRSDEEngine:
    """
    Load EMA weights from the checkpoint directly into the base network.
    """
    print(f"Loading checkpoint from {ckpt_path}...")
    
    # 1. Load the model with strict=False to avoid errors if EMA keys are unmapped
    model = IRSDEEngine.load_from_checkpoint(ckpt_path, strict=False)
    
    # 2. Load the raw checkpoint dictionary
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", {})
    
    # 3. Extract EMA weights
    ema_state_dict = {}
    for k, v in state_dict.items():
        if ema_prefix in k:
            # Remove the prefix to match the structural keys of model.net
            # e.g., "net_ema.layer1.weight" -> "layer1.weight"
            new_key = k.replace(ema_prefix, "")
            ema_state_dict[new_key] = v
            
    # 4. Inject EMA weights into the base network
    if ema_state_dict:
        print(f"[*] Successfully extracted {len(ema_state_dict)} EMA parameters.")
        # Load the EMA weights directly into the primary network
        model.net.load_state_dict(ema_state_dict, strict=True)
        print("[*] EMA weights loaded into model.net for inference.")
    else:
        print("[!] No EMA weights found with prefix '{}'. Using standard weights.".format(ema_prefix))
        
    return model

def run_inference_and_save_steps(
    ckpt_path: str, 
    lr_img_path: str, 
    save_dir: str = "./inference_results", 
    num_states: int = 8
):
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(lr_img_path).split('.')[0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # === UPDATED: Use the custom EMA loader ===
    # Assuming your EMACallback saves weights with "net_ema." prefix
    model = load_ema_weights(model=IRSDEEngine, ckpt_path=ckpt_path, ema_prefix="net_ema.")
    
    model.eval()
    model.to(device)
    
    # Load and preprocess image
    img = Image.open(lr_img_path).convert('RGB')
    transform = T.ToTensor()
    lr_tensor = transform(img).unsqueeze(0).to(device) 
    
    print("Starting reverse SDE inference...")
    with torch.no_grad():
        # Now model.net contains the EMA weights
        model.sde.set_model(model.net)
        
        h, w = lr_tensor.shape[-2], lr_tensor.shape[-1]
        target_shape = (h * model.scale_factor, w * model.scale_factor)
        
        mu = model._upscale_lr(lr_tensor, target_shape)
        model.sde.set_mu(mu)
        
        xt = model.sde.noise_state(mu)
        preds, intermediate_states = model.reverse_sde_with_intermediates(xt, num_states=num_states)
    
    # Save step-by-step visualizations
    vutils.save_image(mu, os.path.join(save_dir, f"{base_name}_00_mu_upscaled.png"))
    vutils.save_image(xt, os.path.join(save_dir, f"{base_name}_01_xt_noise.png"))
    
    for i, state in enumerate(intermediate_states):
        step_idx = i + 2
        file_path = os.path.join(save_dir, f"{base_name}_{step_idx:02d}_denoise_step.png")
        vutils.save_image(state, file_path)
        
    final_path = os.path.join(save_dir, f"{base_name}_{len(intermediate_states)+2:02d}_final_pred.png")
    vutils.save_image(preds, final_path)
    
    print("Inference complete.")

if __name__ == "__main__":
    # Example usage:
    # Adjust these paths to your actual local files
    CHECKPOINT_FILE = "logs/EDiffIR/X4/version_8/epoch=599-last.ckpt"
    INPUT_IMAGE = "/data/RiceSR2024/RGB-SAR/dataset/test/LR_bicubic/X4/Xipo1-Align-Fill_5760_1920.png" 
    OUTPUT_FOLDER = "logs/EDiffIR/X4/sde_visualization"
    
    run_inference_and_save_steps(
        ckpt_path=CHECKPOINT_FILE,
        lr_img_path=INPUT_IMAGE,
        save_dir=OUTPUT_FOLDER,
        num_states=10 # Extract 10 intermediate frames during reverse diffusion
    )