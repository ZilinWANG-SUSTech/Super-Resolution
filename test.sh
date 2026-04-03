# python test.py \
# --config configs/ldm/bsr_sr_ae_kl_64x64x3.yaml \
# --ckpt checkpoints/LDM/latent_diffusion/best-epoch=266-val_psnr=24.1217.ckpt


# python test.py \
# --config configs/DiffIR/DiffIRS2-GAN.yaml \
# --ckpt checkpoints/DiffIR/S2-GAN/best-epoch=008-val_psnr=26.0110.ckpt

# python test.py \
# --config configs/swinir/swinir-classical.yaml \
# --ckpt checkpoints/SwinIR-V2/best-epoch=080-val_psnr=26.1691.ckpt

# python test.py \
# --config configs/mambair/mambair.yaml \
# --ckpt checkpoints/MambaIR-v2/best-epoch=044-val_psnr=26.1289.ckpt

# SwinIR
python test.py \
--config configs/swinir/swinir-classical-x16.yaml \
--ckpt logs/SwinIR-x16/version_0/best-epoch=095-val/psnr=24.8028.ckpt \
-n SwinIR-x16

# python train.py \
# --config configs/swinir/swinir-classical-x4.yaml \
# -n SwinIR-x4

# python train.py \
# --config configs/swinir/swinir-classical-x8.yaml \
# -n SwinIR-x8