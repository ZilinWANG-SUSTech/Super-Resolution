# python test.py \
# --config configs/ldm/bsr_sr_ae_kl_64x64x3.yaml \
# --ckpt checkpoints/LDM/latent_diffusion/best-epoch=266-val_psnr=24.1217.ckpt


# python test.py \
# --config configs/DiffIR/DiffIRS2-GAN.yaml \
# --ckpt checkpoints/DiffIR/S2-GAN/best-epoch=008-val_psnr=26.0110.ckpt

python test.py \
--config configs/swinir/swinir-classical.yaml \
--ckpt checkpoints/SwinIR/best-epoch=014-val_psnr=26.1698.ckpt
