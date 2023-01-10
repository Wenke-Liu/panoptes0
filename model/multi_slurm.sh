#!/bin/bash
label='Arm1q'
date='230106'

for model in X1 X2 X3 X4 F1 F2 F3 F4;
do 

mv "./test_scripts/test_${model}_${label}.sh" .
sbatch "test_${model}_${label}.sh"
mv "test_${model}_${label}.sh" "../results/NL5_${label}/${model}/${date}"
done

for model in X1 X2 X3 X4 F1 F2 F3 F4;
do 
mv "../results/NL5_${label}/${model}/train_panoptes.sh" "../results/NL5_${label}/${model}/${date}/train_panoptes_${date}.sh"
done
