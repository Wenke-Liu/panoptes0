import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from math import ceil
from tensorflow.keras.utils import to_categorical
from data_input import DataSet
from model import PANOPTES
from utils import *
from sklearn.manifold import TSNE
from sklearn import metrics

parser = argparse.ArgumentParser()
parser.add_argument('--multi_gpu', type=bool, default=True,
                    help='Boolean. Whether to use multi-gpu training.')
parser.add_argument('--tile_idx_dir', type=str, default='idx_files/pancan_imaging_all_tiles_index_TN.csv',
                    help='Path to tile index file.')
parser.add_argument('--label_df_dir', type=str, default='lab_files/pancan_1.1_ps_gene_drug_label.csv',
                    help='Path to label index file.')
parser.add_argument('--lab_col', type=str, default=None, help='Name of the label column in the data table.')
parser.add_argument('--tumor_only', type=bool, default=True, help='Whether to use tumor slides only.')
parser.add_argument('--tumor_type', type=str, nargs='+', default=None,
                    help='List of selected cancer types.')
parser.add_argument('--out_dir', type=str, default='./results/test', help='Parent output directory.')
parser.add_argument('--val_sample', type=int, default=10000, help='Number of validation examples.')
parser.add_argument('--manifold_sample', type=int, default=20000,
                    help='Number of test examples used to estimate manifold.')
parser.add_argument('--batch_size', type=int, default=16, help='Training and testing batch size.')
parser.add_argument('--steps', type=int, default=10000,
                    help='Number of steps per epoch, if smaller than the maximum steps per epoch.')
parser.add_argument('--max_epoch', type=int, default=20, help='Maximum number of epochs trained.')
parser.add_argument('--patience', type=int, default=2, help='Patience during training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for data split and sampling.')

args = parser.parse_args()

"""
Global variables, use argparse when run in cli
"""
MULTI_GPU = args.multi_gpu
TILE_IDX_DIR = args.tile_idx_dir
LABEL_DF_DIR = args.label_df_dir
LAB_COL = args.lab_col
TUMOR_ONLY = args.tumor_only
TUMOR_TYPE = args.tumor_type
OUT_DIR = args.out_dir
VAL_SAMPLE = args.val_sample
MANIFOLD_SAMPLE = args.manifold_sample
BATCH_SIZE = args.batch_size
STEPS = args.steps
MAX_EPOCH = args.max_epoch
PATIENCE = args.patience
SEED = args.seed

print('Experiment random seed: ' + str(SEED))
np.random.seed(SEED)
SPLIT_SEED, VAL_SEED, TST_SEED = np.random.randint(low=0, high=1000000, size=3)
print('Data split seed: ' + str(SPLIT_SEED))
print('Validation sample seed: ' + str(VAL_SEED))
print('Manifold sample seed: ' + str(TST_SEED))

try:
    os.makedirs(OUT_DIR + '/data')
    os.makedirs(OUT_DIR + '/pred')
except FileExistsError:
    pass

data_idx = pd.read_csv(TILE_IDX_DIR)
data_idx = data_idx.dropna()

if TUMOR_ONLY:
    data_idx = data_idx.loc[data_idx['Tumor_Normal'] == 'tumor']

if TUMOR_TYPE is not None:
    print('Using data from: ' + str(TUMOR_TYPE))
    data_idx = data_idx.loc[data_idx['Tumor'].isin(TUMOR_TYPE)]

lab_df = pd.read_csv(LABEL_DF_DIR)
data_idx = data_idx.merge(lab_df[['Patient_ID', LAB_COL]], how='left')  # can change
print('Renaming column {} into label column.'.format(LAB_COL))
data_idx = data_idx.rename(columns={LAB_COL: 'label'})

data_idx = data_idx.dropna()

print('All data idx: {}'.format(str(data_idx.shape)))

trn_id, val_id, tst_id = data_split(data_idx,
                                    split_ratio=(0.7, 0.15, 0.15),
                                    stratify='label')

trn_df = data_idx.loc[data_idx['Patient_ID'].isin(trn_id)].copy()
trn_df = stratified_weights(trn_df)
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=trn_df, fn='trn')

print('Number of training examples: ' + str(trn_df.shape[0]))
MAX_STEPS = ceil(trn_df.shape[0]/BATCH_SIZE)    # calculate steps in each epoch
print('Maximum steps per epoch: ' + str(MAX_STEPS))

val_df = data_idx.loc[data_idx['Patient_ID'].isin(val_id)].copy()
val_df = val_df.sample(n=VAL_SAMPLE, random_state=VAL_SEED)    # sample for efficient validation
val_df['sample_weights'] = 1
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=val_df, fn='val')

tst_df = data_idx.loc[data_idx['Patient_ID'].isin(tst_id)].copy()
#tst_df = tst_df.sample(n=VAL_SAMPLE, random_state=SEED)    # sample
tst_df['sample_weights'] = 1
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=tst_df, fn='tst')

trn = DataSet(filenames=trn_df[['L1path', 'L2path', 'L3path']],
              labels=to_categorical(trn_df['label']), 
              tile_weights=trn_df['sample_weights'], id_level=3)

val = DataSet(filenames=val_df[['L1path', 'L2path', 'L3path']],
              labels=to_categorical(val_df['label']), 
              tile_weights=val_df['sample_weights'], id_level=3)

tst = DataSet(filenames=tst_df[['L1path', 'L2path', 'L3path']],
              labels=to_categorical(tst_df['label']), 
              tile_weights=tst_df['sample_weights'], id_level=3)

trn_ds = trn.create_dataset(batch_size=BATCH_SIZE, ds_epoch=None)
val_ds = val.create_dataset(batch_size=BATCH_SIZE, ds_epoch=1)
tst_ds = tst.create_dataset(shuffle=False, batch_size=BATCH_SIZE, ds_epoch=1)


if MULTI_GPU:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = PANOPTES()
        #model.compile(loss_fn=tf.keras.losses.SparseCategoricalCrossentropy())

else:
    model = PANOPTES()
    #model.compile(loss_fn=tf.keras.losses.CategoricalCrossentropy())

model.train(trn_data=trn_ds, val_data=val_ds,
            steps=min(STEPS, MAX_STEPS),
            patience=PATIENCE,
            n_epoch=MAX_EPOCH,
            log_dir=OUT_DIR + '/log',
            model_dir=OUT_DIR + '/model')

model.print_attr()

tst_res = model.inference(tst_ds)


tst_df['NEG_Score'] = tst_res[1][:, 0]
tst_df['POS_Score'] = tst_res[1][:, 1]
tst_df.to_csv(OUT_DIR + '/pred/tst_tile_pred.csv', index=False)

tst_df_pred = tst_df[['Patient_ID', 'Slide_ID', 'Tumor', 'Tumor_Normal', 'label', 'NEG_Score', 'POS_Score']]
tst_df_pred_slide = tst_df_pred.groupby(['Patient_ID', 'Slide_ID', 'Tumor', 'Tumor_Normal']).agg('mean').reset_index()
tst_df_pred_slide.to_csv(OUT_DIR + '/pred/tst_slide_pred.csv', index=False)

fpr, tpr, thresholds = metrics.roc_curve(tst_df_pred_slide['label'], tst_df_pred_slide['POS_Score'], pos_label=1)
print('Slide level AUROC on test data: ' +
      str(metrics.auc(fpr, tpr)))

np.random.seed(TST_SEED)
sample_idx = np.random.choice(tst_df.shape[0], MANIFOLD_SAMPLE)    # sample 20,000 tiles for TSNE
print(tst_res[0].shape)
tst_sampled = tst_res[0][sample_idx, :]
tst_df_sampled = tst_df.copy()
tst_df_sampled = tst_df_sampled.iloc[sample_idx, :]

tsne_embedding = TSNE(n_components=2).fit_transform(tst_sampled)

tst_df_sampled['tsne_0'] = tsne_embedding[:, 0]
tst_df_sampled['tsne_1'] = tsne_embedding[:, 1]
tst_df_sampled['NEG_Score'] = tst_res[1][sample_idx, 0]
tst_df_sampled['POS_Score'] = tst_res[1][sample_idx, 1]

tst_df_sampled.to_csv(OUT_DIR + '/pred/tSNE_P_N.csv', index=False)

