#!/bin/bash
#
#SBATCH --job-name=panoptes_tf2
#SBATCH --partition=gpu8_long,gpu4_long
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --begin=now
#SBATCH --time=8-00:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --mail-user=wenke.liu@nyulangone.org
#SBATCH --output=tst_panoptes_%j.out
#SBATCH --error=tst_panotpes_%j.error
#SBATCH --gres=gpu:1

module load condaenvs/gpu/tensorflow2.2
#module load anaconda3/gpu/5.2.0-cuda10.1
#conda activate tensorflow2.5

python test.py \
    --out_dir='../results/NL5_POLE/F2/dropout7/test' \
    --tst_df='../idx_files/ucec_conf_tile_idx.csv' \
    --lab_col='subtype_POLE' \
    --covariate=Age,BMI \
    --base_model='InceptionResNetV1' \
    --saved_model_dir='../results/NL5_POLE/F2/dropout7/model/panoptes_weights_final.h5' \
    --feature_pool=False \
    --aux=True \
    --seed=221207