export HF_ENDPOINT=https://hf-mirror.com


#######     SwinIR  ############
# python inference.py \
# tsp ./run_inference.sh \
# --config configs/swinir/swinir-classical-x16.yaml \
# -n SwinIR-x16 \
# --ckpt logs/SwinIR-x16/version_0/best-epoch=098-val/psnr=24.8009.ckpt


# python inference.py \
# tsp ./run_inference.sh \
# --config configs/swinir/swinir-classical-x4.yaml \
# -n SwinIR-x4 \
# --ckpt logs/SwinIR-x4/version_0/best-epoch=299-val/psnr=25.9014.ckpt

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/swinir/swinir-classical-x8.yaml \
# -n SwinIR-x8 \
# --ckpt logs/SwinIR-x8/version_0/best-epoch=059-val/psnr=25.2266.ckpt


#######     DiffIR  ############

##### S1
# python inference.py \
# --config configs/DiffIR/DiffIRS1-X16.yaml \
# -n DiffIR/S1/DiffIR-x16

# python inference.py \
# --config configs/DiffIR/DiffIRS1-X4.yaml \
# -n DiffIR/S1/DiffIR-x4

# python inference.py \
# --config configs/DiffIR/DiffIRS1-X8.yaml \
# -n DiffIR/S1/DiffIR-x8


##### S2
### TODO: Add S1 Checkpoint
# tsp ./run_inference.sh python inference.py \
# --config configs/DiffIR/DiffIRS2-X16.yaml \
# -n DiffIR/S2/DiffIR-x16

# tsp ./run_inference.sh \
# --config configs/DiffIR/DiffIRS2-X4.yaml \
# -n DiffIR/S2/DiffIR-x4

# tsp ./run_inference.sh \
# --config configs/DiffIR/DiffIRS2-X8.yaml \
# -n DiffIR/S2/DiffIR-x8

#### GAN 
# python inference.py \
# tsp ./run_inference.sh \
# --config configs/DiffIR/DiffIRS2-GAN-X4.yaml \
# -n DiffIR/S2-GAN/DiffIR-x4 \
# --ckpt logs/DiffIR/S2-GAN/DiffIR-x4/version_0/best-epoch=002-val/psnr=25.8149.ckpt


# python inference.py \
# tsp ./run_inference.sh \
# --config configs/DiffIR/DiffIRS2-GAN-X16.yaml \
# -n DiffIR/S2-GAN/DiffIR-x16 \
# --ckpt logs/DiffIR/S2-GAN/DiffIR-x16/version_0/best-epoch=002-val/psnr=24.7007.ckpt


# python inference.py \
# tsp ./run_inference.sh \
# --config configs/DiffIR/DiffIRS2-GAN-X8.yaml \
# -n DiffIR/S2-GAN/DiffIR-x8 \
# --ckpt logs/DiffIR/S2-GAN/DiffIR-x8/version_0/best-epoch=014-val/psnr=25.0510.ckpt



#######     MambaIRv2  ############
# python inference.py \
# tsp ./run_inference.sh \
# --config configs/mambairv2/mambairv2-x16.yaml \
# -n MambaIRv2/MambaIRv2-x16 \
# --ckpt logs/MambaIRv2/MambaIRv2-x16/version_0/best-epoch=032-val/psnr=24.8344.ckpt

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/mambairv2/mambairv2-x4.yaml \
# -n MambaIRv2/MambaIRv2-x4 \
# --ckpt logs/MambaIRv2/MambaIRv2-x4/version_0/best-epoch=062-val/psnr=25.9928.ckpt

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/mambairv2/mambairv2-x8.yaml \
# -n MambaIRv2/MambaIRv2-x8 \
# --ckpt logs/MambaIRv2/MambaIRv2-x8/version_0/best-epoch=047-val/psnr=25.2182.ckpt


#######     RealESRGAN  ############

#### Regression
# python inference.py \
# tsp ./run_inference.sh \
# --config configs/realesrgan/realesrnet-x4.yaml \
# -n RealESRGAN/Regression/X4 \
# --resume logs/RealESRGAN/Regression/X4/version_1/epoch=020-last.ckpt

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/realesrgan/realesrnet-x8.yaml \
# -n RealESRGAN/Regression/X8 \
# --resume logs/RealESRGAN/Regression/X8/version_0/epoch=056-last.ckpt

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/realesrgan/realesrnet-x16.yaml \
# -n RealESRGAN/Regression/X16

#### GAN
# # python inference.py \
# tsp ./run_inference.sh \
# --config configs/realesrgan/realesrgan-x4.yaml \
# -n RealESRGAN/GAN/X4 \
# --ckpt logs/RealESRGAN/GAN/X4/version_0/best-epoch=002-val/psnr=25.8070.ckpt

# # python inference.py \
# tsp ./run_inference.sh \
# --config configs/realesrgan/realesrgan-x8.yaml \
# -n RealESRGAN/GAN/X8 \
# --ckpt logs/RealESRGAN/GAN/X8/version_0/best-epoch=002-val/psnr=24.8503.ckpt

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/realesrgan/realesrgan-x16.yaml \
# -n RealESRGAN/GAN/X16 \
# --ckpt logs/RealESRGAN/GAN/X16/version_0/best-epoch=002-val/psnr=24.4222.ckpt




#######     ResShift  ############
# #### VQGAN
# tsp ./run_inference.sh \
# python inference.py \
# --config configs/VQGAN/vqgan-x4.yaml \
# -n VQGAN/X4 \
# --ckpt logs/VQGAN/X4/version_12/best-epoch=092-val/psnr=27.0352.ckpt \
# --output_dir logs/VQGAN/X4/inference_results_best_ckpt

# # python inference.py \
# tsp ./run_inference.sh \
# python inference.py \
# --config configs/VQGAN/vqgan-x8.yaml \
# -n VQGAN/X8 \
# --ckpt logs/VQGAN/X8/version_5/epoch=284-last.ckpt \
# --output_dir logs/VQGAN/X8/inference_results_last_ckpt

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/VQGAN/vqgan-x16.yaml \
# -n VQGAN/X16

#### ResShift
# python inference.py \
# tsp ./run_inference.sh \
# python inference.py \
# --config configs/resshift/resshift-x4.yaml \
# -n ResShift/X4 \
# --ckpt logs/ResShift/X4/version_1/best-epoch=155-val/psnr=25.0476.ckpt

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/resshift/resshift-x8.yaml \
# -n ResShift/X8 \
# --ckpt logs/ResShift/X8/version_0/best-epoch=023-val/psnr=24.5291.ckpt

# # python inference.py \
# tsp ./run_inference.sh \
# --config configs/resshift/resshift-x16.yaml \
# -n ResShift/X16 \
# --ckpt logs/ResShift/X16/version_0/best-epoch=170-val/psnr=24.4969.ckpt




#######     UGSR  ############

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/UGSR/ugsr-x4.yaml \
# -n UGSR/X4 \
# --resume logs/UGSR/X4/version_0/epoch=011-last.ckpt

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/UGSR/ugsr-x8.yaml \
# -n UGSR/X8

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/UGSR/ugsr-x16.yaml \
# -n UGSR/X16




#######     OGSRN  ############

#### 1. inference SORTN
# python inference.py \
# tsp ./run_inference.sh \
# --config configs/ogsrn/sortn.yaml \
# -n OGSRN/SORTN \
# --resume logs/OGSRN/SORTN/version_0/epoch=070-last.ckpt


# #### 2. inference SRUN
# python inference.py \
# tsp ./run_inference.sh \
# --config configs/ogsrn/srun-x4.yaml \
# -n OGSRN/SRUN/X4 \
# --ckpt logs/OGSRN/SRUN/X4/version_0/best-epoch=122-val/psnr=26.0314.ckpt

# python inference.py \
# tsp ./run_inference.sh \
# --config configs/ogsrn/srun-x8.yaml \
# -n OGSRN/SRUN/X8 \
# --ckpt logs/OGSRN/SRUN/X8/version_0/best-epoch=068-val/psnr=25.2672.ckpt


# # python inference.py \
# tsp ./run_inference.sh \
# --config configs/ogsrn/srun-x16.yaml \
# -n OGSRN/SRUN/X16 \
# --ckpt logs/OGSRN/SRUN/X16/version_0/best-epoch=260-val/psnr=24.8504.ckpt


#######     EDiffIR  ############

# python inference.py \
tsp ./run_inference.sh \
--config configs/EDiffIR/ediffir-x4.yaml \
-n EDiffIR/X4 \
--ckpt logs/EDiffIR/X4/version_8/epoch=599-last.ckpt
# --ckpt logs/EDiffIR/X4/version_8/best-epoch=029-val/psnr=23.0761.ckpt


# python inference.py \
tsp ./run_inference.sh \
--config configs/EDiffIR/ediffir-x8.yaml \
-n EDiffIR/X8 \
--ckpt logs/EDiffIR/X8/version_2/epoch=359-last.ckpt


tsp ./run_inference.sh \
--config configs/EDiffIR/ediffir-x16.yaml \
-n EDiffIR/X16 \
--ckpt 