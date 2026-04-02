# python predict.py \
# -c configs/DiffIR/DiffIRS2.yaml \
# --ckpt checkpoints/DiffIR/S2/best-epoch=035-val_psnr=26.1441.ckpt \
# --input_dir /data/RiceSR2024/SAR/dataset/test/LR_bicubic/X4 \
# --output_dir /data/RiceSR2024/Data_Processing/inference/SAR/DiffIR/S2/predict-test_dataset-V2

# python predict.py \
# -c configs/ldm/bsr_sr_ae_kl_64x64x3.yaml \
# --ckpt checkpoints/LDM/latent_diffusion/best-epoch=266-val_psnr=24.1217.ckpt \
# --input_dir /data/RiceSR2024/SAR/dataset/train/LR_bicubic/X4 \
# --output_dir /data/RiceSR2024/Data_Processing/inference/SAR/LDM/predict-train_dataset

# python predict.py \
# -c configs/DiffIR/DiffIRS2.yaml \
# --ckpt checkpoints/DiffIR/S2/best-epoch=035-val_psnr=26.1441.ckpt \
# --input_dir /data/RiceSR2024/Set5/LRbicx4 \
# --output_dir /data/RiceSR2024/Data_Processing/inference/Set5/DiffIR/S2/predict-test_dataset

# python predict.py \
# -c configs/ldm/bsr_sr_ae_kl_64x64x3.yaml \
# --ckpt checkpoints/LDM/latent_diffusion/best-epoch=266-val_psnr=24.1217.ckpt \
# --input_dir /data/RiceSR2024/Set5/LRbicx4 \
# --output_dir /data/RiceSR2024/Data_Processing/inference/Set5/LDM/predict-test_dataset


# predict train
# python predict.py \
# -c configs/swinir/swinir-classical.yaml \
# --ckpt checkpoints/SwinIR-V2/best-epoch=080-val_psnr=26.1691.ckpt \
# --input_dir /data/RiceSR2024/SAR/dataset/train/LR_bicubic/X4 \
# --output_dir /data/RiceSR2024/Data_Processing/inference/SAR/SwinIR-V2/predict-train_dataset

# predict test
# python predict.py \
# -c configs/swinir/swinir-classical.yaml \
# --ckpt checkpoints/SwinIR-V2/best-epoch=080-val_psnr=26.1691.ckpt \
# --input_dir /data/RiceSR2024/SAR/dataset/test/LR_bicubic/X4 \
# --output_dir /data/RiceSR2024/Data_Processing/inference/SAR/SwinIR-V2/predict-test_dataset

# predict set5
# python predict.py \
# -c configs/swinir/swinir-classical.yaml \
# --ckpt checkpoints/SwinIR-V2/best-epoch=080-val_psnr=26.1691.ckpt \
# --input_dir /data/RiceSR2024/Set5/LRbicx4 \
# --output_dir /data/RiceSR2024/Data_Processing/inference/Set5/SwinIR-V2/predict-test_dataset

# predict train
# python predict.py \
# -c configs/mambair/mambair.yaml \
# --ckpt checkpoints/MambaIR-v2/best-epoch=044-val_psnr=26.1289.ckpt \
# --input_dir /data/RiceSR2024/SAR/dataset/train/LR_bicubic/X4 \
# --output_dir /data/RiceSR2024/Data_Processing/inference/SAR/MambaIR-V2/predict-train_dataset

# predict test
# python predict.py \
# -c configs/mambair/mambair.yaml \
# --ckpt checkpoints/MambaIR-v2/best-epoch=044-val_psnr=26.1289.ckpt \
# --input_dir /data/RiceSR2024/SAR/dataset/test/LR_bicubic/X4 \
# --output_dir /data/RiceSR2024/Data_Processing/inference/SAR/MambaIR-V2/predict-test_dataset

# # predict set5
python predict.py \
-c configs/mambair/mambair.yaml \ 
--ckpt checkpoints/MambaIR/best-epoch=020-val_psnr=26.1666.ckpt \
--input_dir /data/RiceSR2024/Set5/LRbicx4 \
--output_dir /data/RiceSR2024/Data_Processing/inference/Set5/MambaIR/predict-test_dataset
