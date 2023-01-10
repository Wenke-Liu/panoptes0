#!/bin/bash
label='Arm1q'
labcol='Arm1q_gain'
date='230106'

for model in F1 F2 F3 F4;
do 
cat > test_${model}_${label}.sh <<EOF
#!/bin/bash
#SBATCH --job-name=panoptes_tf2
#SBATCH --partition=gpu8_dev,gpu4_dev
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --begin=now
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --mail-user=wenke.liu@nyulangone.org
#SBATCH --output=${label}${model}_%j.out
#SBATCH --error=${label}${model}_%j.error
#SBATCH --gres=gpu:1

module load condaenvs/gpu/tensorflow2.2

nohup python test.py \\
    --out_dir="../results/NL5_${label}/${model}/${date}/test" \\
    --tst_df='../idx_files/ucec_conf_tile_idx.csv' \\
    --lab_col=${labcol} \\
    --covariate=Age,BMI \\
    --variant=${model} \\
    --saved_model_dir="../results/NL5_${label}/${model}/${date}/model/panoptes_weights_final.h5" \\
    --aux=True \\
    --seed=230106 \\
EOF

mv test_${model}_${label}.sh test_scripts
done

for model in X1 X2 X3 X4;
do 
cat > test_${model}_${label}.sh <<EOF
#!/bin/bash
#SBATCH --job-name=panoptes_tf2
#SBATCH --partition=gpu8_dev,gpu4_dev
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --begin=now
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --mail-user=wenke.liu@nyulangone.org
#SBATCH --output=${label}${model}_%j.out
#SBATCH --error=${label}${model}_%j.error
#SBATCH --gres=gpu:1

module load condaenvs/gpu/tensorflow2.2

nohup python test.py \\
    --out_dir="../results/NL5_${label}/${model}/${date}/test" \\
    --tst_df='../idx_files/ucec_conf_tile_idx.csv' \\
    --lab_col=${labcol} \\
    --variant=${model} \\
    --saved_model_dir="../results/NL5_${label}/${model}/${date}/model/panoptes_weights_final.h5" \\
    --aux=True \\
    --seed=230106 \\
EOF

mv test_${model}_${label}.sh test_scripts
done





