# training with full tile list, label list and data split
nohup python train.py \
    --split=True \
    --out_dir='../results/NL5_POLE/X4/230117' \
    --tile_idx_dir='../idx_files/ucec_all_he_idx.csv' \
    --label_df_dir='../idx_files/ucec_trn_labels_tcga_disc_nyu.csv' \
    --lab_col='POLE' \
    --covariate=Age,BMI \
    --variant=X4 \
    --max_epoch=20 \
    --batch_size=16 \
    --dropout=0.7 \
    --aux=True \
    --aux_weight=0.3 \
    --patience=2 \
    --seed=230117 \
    > 230117_nl5_pole_x4_log.txt 2>&1 &


# training with given data splits
nohup python train.py \
    --split=False \
    --out_dir='../results/NL5_POLE/F2/221220' \
    --trn_df='../idx_files/NL5_POLE/tr_sample_tf2.csv' \
    --val_df='../idx_files/NL5_POLE/va_sample_tf2.csv' \
    --tst_df='../idx_files/NL5_POLE/te_sample_tf2.csv' \
    --covariate=age,BMI \
    --max_epoch=20 \
    --batch_size=16 \
    --dropout=0.7 \
    --base_model='InceptionResNetV1' \
    --feature_pool=False \
    --aux=True \
    --aux_weight=0.5 \
    --patience=5 \
    --seed=221216 \
    > 221220_nl5_pole_f2_log.txt 2>&1 &

nohup python train.py \
    --split=False \
    --out_dir='../results/NL5_POLE/X4/230106' \
    --trn_df='../idx_files/NL5_POLE/tr_sample_tf2.csv' \
    --val_df='../idx_files/NL5_POLE/va_sample_tf2.csv' \
    --tst_df='../idx_files/NL5_POLE/te_sample_tf2.csv' \
    --max_epoch=20 \
    --batch_size=16 \
    --covariate=age,BMI \
    --dropout=0.7 \
    --base_model='InceptionResNetV1' \
    --feature_pool=True \
    --aux=True \
    --aux_weight=0.1 \
    --patience=5 \
    --seed=221216 \
    > 230105_nl5_pole_x4_log.txt 2>&1 &


# loading a saved model and inference on external test set
# make sure to use the same flags as training 
# combined test set: pole_conf_nyu_combined_idx.csv
# CPTAC conf test set: ucec_conf_tile_idx.csv

nohup python test.py \
    --out_dir='../results/NL5_POLE/F2/copied/conf_test' \
    --tst_df='../idx_files/ucec_conf_tile_idx.csv' \
    --lab_col='subtype_POLE' \
    --covariate=Age,BMI \
    --base_model='InceptionResNetV1' \
    --saved_model_dir='../results/copied/NL5F2POLE_weights.h5' \
    --feature_pool=False \
    --aux=True \
    --seed=221216 \
    --legacy=True \
    > 230105_copied_nl5_pole_f2_test_log.txt 2>&1 &


# loading a saved model ckpt different from the final version
nohup python test.py \
    --out_dir='../results/NL5_POLE/F2/221215/test_epoch10' \
    --tst_df='../idx_files/ucec_conf_tile_idx.csv' \
    --lab_col='subtype_POLE' \
    --covariate=Age,BMI \
    --base_model='InceptionResNetV1' \
    --saved_model_dir='../results/NL5_POLE/F2/221215/model/ckpt/weights.010-0.8949.hdf5' \
    --feature_pool=False \
    --aux=True \
    --seed=221215 \
    > 221215_nl5_pole_f2_test_epoch10_log.txt 2>&1 &


# loading a saved model ckpt different from the final version
nohup python test.py \
    --out_dir='../results/NL5_POLE/X4/copied' \
    --tst_df='../idx_files/pole_conf_nyu_combined_idx.csv' \
    --lab_col='subtype_POLE' \
    --base_model='InceptionResNetV1' \
    --saved_model_dir='../results/copied/NL5X4POLE_weights.h5' \
    --feature_pool=True \
    --aux=True \
    --seed=230106 \
    --legacy=True \
    > 230106_copied_nl5_pole_x4_nyu_test_log.txt 2>&1 &
