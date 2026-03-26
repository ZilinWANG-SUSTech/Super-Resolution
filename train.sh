# python train.py \
# --config configs/ldm/autoencoder_kl_64x64x3.yaml \
# -n LDM/autoencoder_kl_64x64x3


# python train.py \
# --config configs/ldm/bsr_sr_ae_kl_64x64x3.yaml \
# -n LDM/latent_diffusion

# python train.py \
# --config configs/swinir/swinir-classical.yaml \
# -n SwinIR

python train.py \
--config configs/mambair/mambair.yaml \
-n MambaIR
