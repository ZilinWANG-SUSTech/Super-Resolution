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

tsp ./run_train.sh \
--config configs/DiffIR/DiffIRS2-X4.yaml \
-n DiffIR/S2/DiffIR-x4

tsp ./run_train.sh \
--config configs/DiffIR/DiffIRS2-X8.yaml \
-n DiffIR/S2/DiffIR-x8



#######     MambaIRv2  ############
# python train.py \
# --config configs/mambairv2/mambairv2-x16.yaml \
# -n MambaIRv2/MambaIRv2-x16

# python train.py \
tsp ./run_train.sh \
--config configs/mambairv2/mambairv2-x4.yaml \
-n MambaIRv2/MambaIRv2-x4

# python train.py \
tsp ./run_train.sh \
--config configs/mambairv2/mambairv2-x8.yaml \
-n MambaIRv2/MambaIRv2-x8

