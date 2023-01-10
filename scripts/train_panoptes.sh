#!/bin/bash
#
#SBATCH --job-name=panoptes_tf2
#SBATCH --partition=gpu8_long,gpu4_long
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --begin=now
#SBATCH --time=8-00:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --mail-user=wenke.liu@nyulangone.org
#SBATCH --output=pik3r1_f2_%j.out
#SBATCH --error=pik3r1_f2_%j.error
#SBATCH --gres=gpu:2

module load condaenvs/gpu/tensorflow2.2

python train.py \
    --multi_gpu=True \
    --split=True \
    --out_dir='../results/NL5_PIK3R1/F2/230109' \
    --tile_idx_dir='../idx_files/ucec_all_he_tiles_index.csv' \
    --label_df_dir='../idx_files/labels_trn_disc_tcga.csv' \
    --lab_col='PIK3R1_in_frame' \
    --covariate=Age,BMI \
    --max_epoch=20 \
    --batch_size=16 \
    --dropout=0.7 \
    --base_model='InceptionResNetV1' \
    --feature_pool=False \
    --aux=True \
    --aux_weight=0.3 \
    --patience=5 \
    --seed=230109





    