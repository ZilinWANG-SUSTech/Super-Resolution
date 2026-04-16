import os
import argparse
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torchvision.utils import save_image
from tqdm import tqdm
from utils import build_engine_cls
from data import build_datamodule
import models
import engines

def parse_args():
    parser = argparse.ArgumentParser(description="Inference Pipeline for RGB-Guided SAR SR")
    parser.add_argument(
        "-c", "--config", type=str, default="configs/default_train.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "-n", "--name", type=str, default="default_exp",
        help="Name of the experiment. Used for defining default output directory."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to the trained checkpoint (.ckpt file)."
    )
    parser.add_argument(
        "--input_dir", type=str, default=None, required=False,
        help="Folder containing LR images. Overrides config test_img_dir."
    )
    parser.add_argument(
        "--guide_dir", type=str, default=None, required=False,
        help="Folder containing RGB guide images. Required if using GuidedSRDataModule."
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, required=False,
        help="Folder to save HR images. Defaults to logs/{name}/inference_results."
    )
    parser.add_argument(
        "--gpus", type=int, nargs="+", default=[0],
        help="GPU indices to use for inference."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = OmegaConf.load(args.config)
    input_key = config.globals.get("input_key", None)

    pl.seed_everything(42, workers=True)
    device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")

    # 2. Setup Directories and DataModule
    data_cfg = OmegaConf.to_container(config.data, resolve=True)
    module_type = data_cfg.get('type', 'SRDataModule')

    # Dynamically override config paths based on the module type
    if module_type == "SRDataModule":
        if args.input_dir:
            data_cfg['test_img_dir'] = args.input_dir
        input_dir = data_cfg.get('test_img_dir')
        
    elif module_type == "GuidedSRDataModule":
        if args.input_dir:
            data_cfg['test_lr_dir'] = args.input_dir
        if args.guide_dir:
            data_cfg['test_guide_dir'] = args.guide_dir
            
        input_dir = data_cfg.get('test_lr_dir')
        guide_dir = data_cfg.get('test_guide_dir')
        
        # Additional validation for guided SR
        if not guide_dir or not os.path.exists(guide_dir):
            raise ValueError(f"Guide directory missing or invalid: {guide_dir}. "
                             "Please provide --guide_dir or set test_guide_dir in config.")
    else:
        raise ValueError(f"Unknown datamodule type: {module_type}")

    output_dir = args.output_dir if args.output_dir else os.path.join("logs", args.name, "inference_results")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Starting Inference ===")
    print(f"DataModule Type:  {module_type}")
    print(f"Input Directory:  {input_dir}")
    if module_type == "GuidedSRDataModule":
        print(f"Guide Directory:  {guide_dir}")
    print(f"Output Directory: {output_dir}")

    # Build DataModule using your factory function
    datamodule = build_datamodule(data_cfg)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # 3. Load Model and EMA Weights
    lightning_cfg = OmegaConf.to_container(config.lightning_module, resolve=True)
    engine_cls = build_engine_cls(lightning_cfg)
    
    # Initialize the model structure without loading weights yet
    module = engine_cls(**lightning_cfg.get('params', {}))
    
    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Search for EMA weights and remap them to the main network
    ema_state_dict = {
        k.replace("net_ema.", "net."): v 
        for k, v in state_dict.items() if k.startswith("net_ema.")
    }

    if ema_state_dict:
        print("Found EMA weights. Injecting them into the main network for inference...")
        # Override the main network weights with EMA weights
        state_dict.update(ema_state_dict)
    else:
        print("No EMA weights found. Using standard network weights.")

    # Load the unified state_dict. strict=False ignores missing 'net_ema' keys in the module
    module.load_state_dict(state_dict, strict=False)
    module.to(device)
    module.eval()

    # 4. Run Inference Loop over DataModule
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
            if input_key:
                lr_tensor = batch[input_key].to(device)
            elif 'img' in batch:
                lr_tensor = batch['img'].to(device)
            elif 'lr' in batch:
                lr_tensor = batch['lr'].to(device)
            else:
                raise KeyError("Batch does not contain 'img' or 'lr' keys.")

            batch_size = lr_tensor.size(0)

            img_names = batch.get('img_name', [f"batch_{batch_idx}_img_{i}.png" for i in range(batch_size)])

            if 'guide' in batch:
                guide_tensor = batch['guide'].to(device)
                sr_tensors = module.inference(lr_tensor, guided_tensor=guide_tensor)
            else:
                sr_tensors = module.inference(lr_tensor)

            for i in range(batch_size):
                save_name = img_names[i]
                save_path = os.path.join(output_dir, save_name)
                save_image(sr_tensors[i], save_path)

    print(f"=== Inference Completed! Results saved to {output_dir} ===")

if __name__ == "__main__":
    main()