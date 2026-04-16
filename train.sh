export HF_ENDPOINT=https://hf-mirror.com


#######     SwinIR  ############
# python train.py \
# --config configs/swinir/swinir-classical-x16.yaml \
# -n SwinIR-x16 \
# --resume logs/SwinIR-x16/version_1/best-epoch=005-val/psnr=20.7549.ckpt

# python train.py \
# --config configs/swinir/swinir-classical-x4.yaml \
# -n SwinIR-x4

# python train.py \
# --config configs/swinir/swinir-classical-x8.yaml \
# -n SwinIR-x8


#######     DiffIR  ############

##### S1
# python train.py \
# --config configs/DiffIR/DiffIRS1-X16.yaml \
# -n DiffIR/S1/DiffIR-x16

# python train.py \
# --config configs/DiffIR/DiffIRS1-X4.yaml \
# -n DiffIR/S1/DiffIR-x4

# python train.py \
# --config configs/DiffIR/DiffIRS1-X8.yaml \
# -n DiffIR/S1/DiffIR-x8


##### S2
### TODO: Add S1 Checkpoint
# tsp ./run_train.sh python train.py \
# --config configs/DiffIR/DiffIRS2-X16.yaml \
# -n DiffIR/S2/DiffIR-x16

# tsp ./run_train.sh \
# --config configs/DiffIR/DiffIRS2-X4.yaml \
# -n DiffIR/S2/DiffIR-x4

# tsp ./run_train.sh \
# --config configs/DiffIR/DiffIRS2-X8.yaml \
# -n DiffIR/S2/DiffIR-x8

#### GAN 
# python train.py \
# tsp ./run_train.sh \
# --config configs/DiffIR/DiffIRS2-GAN-X4.yaml \
# -n DiffIR/S2-GAN/DiffIR-x4

# python train.py \
# tsp ./run_train.sh \
# --config configs/DiffIR/DiffIRS2-GAN-X16.yaml \
# -n DiffIR/S2-GAN/DiffIR-x16

# python train.py \
# tsp ./run_train.sh \
# --config configs/DiffIR/DiffIRS2-GAN-X8.yaml \
# -n DiffIR/S2-GAN/DiffIR-x8



#######     MambaIRv2  ############
# python train.py \
# --config configs/mambairv2/mambairv2-x16.yaml \
# -n MambaIRv2/MambaIRv2-x16

# python train.py \
# tsp ./run_train.sh \
# --config configs/mambairv2/mambairv2-x4.yaml \
# -n MambaIRv2/MambaIRv2-x4

# python train.py \
# tsp ./run_train.sh \
# --config configs/mambairv2/mambairv2-x8.yaml \
# -n MambaIRv2/MambaIRv2-x8


#######     RealESRGAN  ############

#### Regression
# python train.py \
# tsp ./run_train.sh \
# --config configs/realesrgan/realesrnet-x4.yaml \
# -n RealESRGAN/Regression/X4 \
# --resume logs/RealESRGAN/Regression/X4/version_1/epoch=020-last.ckpt

# python train.py \
# tsp ./run_train.sh \
# --config configs/realesrgan/realesrnet-x8.yaml \
# -n RealESRGAN/Regression/X8 \
# --resume logs/RealESRGAN/Regression/X8/version_0/epoch=056-last.ckpt

# python train.py \
# tsp ./run_train.sh \
# --config configs/realesrgan/realesrnet-x16.yaml \
# -n RealESRGAN/Regression/X16

#### GAN
# # python train.py \
# tsp ./run_train.sh \
# --config configs/realesrgan/realesrgan-x4.yaml \
# -n RealESRGAN/GAN/X4

# # python train.py \
# tsp ./run_train.sh \
# --config configs/realesrgan/realesrgan-x8.yaml \
# -n RealESRGAN/GAN/X8

# python train.py \
# tsp ./run_train.sh \
# --config configs/realesrgan/realesrgan-x16.yaml \
# -n RealESRGAN/GAN/X16




#######     ResShift  ############
#### VQGAN
# python train.py \
tsp ./run_train.sh \
--config configs/VQGAN/vqgan-x4.yaml \
-n VQGAN/X4

# python train.py \
tsp ./run_train.sh \
--config configs/VQGAN/vqgan-x8.yaml \
-n VQGAN/X8

# python train.py \
tsp ./run_train.sh \
--config configs/VQGAN/vqgan-x16.yaml \
-n VQGAN/X16



#######     UGSR  ############

# python train.py \
# tsp ./run_train.sh \
# --config configs/UGSR/ugsr-x4.yaml \
# -n UGSR/X4 \
# --resume logs/UGSR/X4/version_0/epoch=011-last.ckpt

# python train.py \
# tsp ./run_train.sh \
# --config configs/UGSR/ugsr-x8.yaml \
# -n UGSR/X8

# python train.py \
# tsp ./run_train.sh \
# --config configs/UGSR/ugsr-x16.yaml \
# -n UGSR/X16




#######     OGSRN  ############

#### 1. Train SORTN
# python train.py \
# tsp ./run_train.sh \
# --config configs/ogsrn/sortn.yaml \
# -n OGSRN/SORTN \
# --resume logs/OGSRN/SORTN/version_0/epoch=070-last.ckpt


#### 2. Train SRUN
# python train.py \
# tsp ./run_train.sh \
# --config configs/ogsrn/srun-x4.yaml \
# -n OGSRN/SRUN/X4

# python train.py \
# tsp ./run_train.sh \
# --config configs/ogsrn/srun-x8.yaml \
# -n OGSRN/SRUN/X8

# python train.py \
# tsp ./run_train.sh \
# --config configs/ogsrn/srun-x16.yaml \
# -n OGSRN/SRUN/X16