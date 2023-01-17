import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from math import ceil
from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
from data_input import DataSet
from model import PANOPTES
from sklearn.manifold import TSNE
from sklearn import metrics

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--multi_gpu', type=str_to_bool, nargs='?', const=True, default=False,
                    help='Boolean. Whether to use multi-gpu training.')
parser.add_argument('--out_dir', type=str, default='./results/test', help='Parent output directory.')
parser.add_argument('--variant', type=str, default=None, help='Model variant abbreviation.')
parser.add_argument('--split', type=str_to_bool, nargs='?', const=True, default=False, 
                    help='Whether to split data or give trn, val and tst data tables.')
parser.add_argument('--tile_idx_dir', type=str, default='idx_files/pancan_imaging_all_tiles_index_TN.csv',
                    help='Path to tile index file.')
parser.add_argument('--label_df_dir', type=str, default='lab_files/pancan_1.1_ps_gene_drug_label.csv',
                    help='Path to label index file.')
parser.add_argument('--trn_df', type=str, default='./', help='Path to trn index file.')  
parser.add_argument('--val_df', type=str, default='./', help='Path to val index file.')  
parser.add_argument('--tst_df', type=str, default='./', help='Path to tst index file.')  
parser.add_argument('--lab_col', type=str, default=None, help='Name of the label column in the label data table.')
parser.add_argument('--covariate', type=str, default=None, help='List of selected covarite columns, comma-delimited.')
parser.add_argument('--val_sample', type=int, default=None, help='Number of validation examples.')
parser.add_argument('--manifold_sample', type=int, default=20000,
                    help='Number of test examples used to estimate manifold.')
parser.add_argument('--base_model', type=str, default='InceptionResNetV1', help='Name of the branch base model.')  
parser.add_argument('--aux', type=str_to_bool, nargs='?', const=True, default=False, help='Whether use auxiliary outputs for training.') 
parser.add_argument('--aux_weight', type=float, default=0., help='Loss weight for auxiliary outputs.') 
parser.add_argument('--feature_pool', type=str_to_bool, nargs='?', const=True, default=False, help='Whether include feature pooling')  
parser.add_argument('--num_class', type=int, default=2, help='Label classes.')
parser.add_argument('--batch_size', type=int, default=16, help='Training and testing batch size.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability.')
parser.add_argument('--steps', type=int, default=10000,
                    help='Number of steps per epoch, if smaller than the maximum steps per epoch.')
parser.add_argument('--max_epoch', type=int, default=20, help='Maximum number of epochs trained.')
parser.add_argument('--patience', type=int, default=2, help='Patience during training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for data split and sampling.')
parser.add_argument('--agg_level', type=str, default='Patient_ID', help='Aggregation level of test inference.')

args = parser.parse_args()
print('command line inputs:')
print(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))

"""
Global variables, use argparse when run in cli
"""
MULTI_GPU = args.multi_gpu
VARIANT = args.variant
COVARIATE = args.covariate
OUT_DIR = args.out_dir
VAL_SAMPLE = args.val_sample
MANIFOLD_SAMPLE = args.manifold_sample
BASE_MODEL = args.base_model
AUX = args.aux
AUX_W = args.aux_weight
NUM_CLASS = args.num_class
DROPOUT = args.dropout
FEATURE_POOL = args.feature_pool
BATCH_SIZE = args.batch_size
STEPS = args.steps
MAX_EPOCH = args.max_epoch
PATIENCE = args.patience
SEED = args.seed
AGG = args.agg_level

if not (VARIANT.startswith(('X', 'F')) and VARIANT.endswith(('1', '2', '3', '4'))):
    print('Variant abbreviation not valid. Using individual flags.')

else:
    if VARIANT.startswith('X'):  
        COVARIATE = None

    if VARIANT.endswith(('1', '3')):
        BASE_MODEL = 'InceptionResNetV2'
    else:
        BASE_MODEL = 'InceptionResNetV1'

    if VARIANT.endswith(('1', '2')):
        FEATURE_POOL = False
    else:
        FEATURE_POOL = True
    
    print('Overridden by abbreviation if different:')    # use abbreviations
    print('covariate: {}'.format(str(COVARIATE)))
    print('base_model: {}'.format(BASE_MODEL))
    print('feature_pool: {}'.format(str(FEATURE_POOL)))

try:
    os.makedirs(OUT_DIR + '/data')
    os.makedirs(OUT_DIR + '/pred')
except FileExistsError:
    pass

print('Experiment random seed: ' + str(SEED))
np.random.seed(SEED)
SPLIT_SEED, VAL_SEED, TST_SEED = np.random.randint(low=0, high=1000000, size=3)
print('Data split seed: ' + str(SPLIT_SEED))
print('Validation sample seed: ' + str(VAL_SEED))
print('Manifold sample seed: ' + str(TST_SEED))


if COVARIATE is not None:
    COVARIATE = COVARIATE.split(',')   # list of covariates 
    print('Using covariates: ' + str(COVARIATE))
    N_COV = len(COVARIATE)

else:
    N_COV = None

if not args.split:
    print('Using given data indices...')
    print('Training data: ' + args.trn_df)
    trn_df = pd.read_csv(args.trn_df)
    val_df = pd.read_csv(args.val_df)
    tst_df = pd.read_csv(args.tst_df)
    print('Training data: ' + args.trn_df)
    print('Validation data: ' + args.val_df)
    print('Testing data: ' + args.tst_df)


else:    # patient level random split controled by SPLIT_SEED
    print('Using all data indices: ' + args.tile_idx_dir)
    data_idx = pd.read_csv(args.tile_idx_dir)
    data_idx = data_idx.dropna()
    print('Using label data: ' + args.label_df_dir)
    lab_df = pd.read_csv(args.label_df_dir)
    data_idx = data_idx.merge(lab_df[['Patient_ID', 'Tumor', args.lab_col]], how='left')  # can change
    print('Renaming column {} into label column.'.format(args.lab_col))
    data_idx = data_idx.rename(columns={args.lab_col: 'label'})

    if COVARIATE is not None:
        print('Using covariates: ' + str(COVARIATE))
        
        data_idx = data_idx.merge(lab_df[COVARIATE + ['Patient_ID']], how='left') 
        for cov in COVARIATE:
            data_idx[cov] = pd.to_numeric(data_idx[cov], errors='coerce')

    data_idx = data_idx.dropna()

    print('All data idx: {}'.format(str(data_idx.shape)))

    trn_id, val_id, tst_id = data_split(data_idx,
                                    split_ratio=(0.8, 0.1, 0.1),
                                    stratify='label',
                                    seed=SPLIT_SEED)
    
    trn_df = data_idx.loc[data_idx['Patient_ID'].isin(trn_id)].copy().reset_index(drop=True)
    
    val_df = data_idx.loc[data_idx['Patient_ID'].isin(val_id)].copy().reset_index(drop=True)
    if VAL_SAMPLE is not None:
        val_df = val_df.sample(n=VAL_SAMPLE, random_state=VAL_SEED).reset_index(drop=True)    # sample
      
    tst_df = data_idx.loc[data_idx['Patient_ID'].isin(tst_id)].copy().reset_index(drop=True)
    

print('Applying training sample weights:')
trn_df = stratified_weights(trn_df)
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=trn_df, fn='trn')

val_df['sample_weights'] = 1  # unweighted
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=val_df, fn='val')

tst_df['sample_weights'] = 1  # unweighted
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=tst_df, fn='tst')


print('Number of training examples: ' + str(trn_df.shape[0]))
MAX_STEPS = ceil(trn_df.shape[0]/BATCH_SIZE)
print('Maximum steps per epoch: ' + str(MAX_STEPS))

if COVARIATE is None:
    trn = DataSet(filenames=trn_df[['L1path', 'L2path', 'L3path']], labels=trn_df['label'], tile_weights=trn_df['sample_weights'])
    val = DataSet(filenames=val_df[['L1path', 'L2path', 'L3path']], labels=val_df['label'], tile_weights=val_df['sample_weights'])
    tst = DataSet(filenames=tst_df[['L1path', 'L2path', 'L3path']], labels=tst_df['label'], tile_weights=tst_df['sample_weights'])

else:
    
    trn = DataSet(filenames=trn_df[['L1path', 'L2path', 'L3path']], 
                  labels=trn_df['label'], covariate=trn_df[COVARIATE], 
                  tile_weights=trn_df['sample_weights'])
    val = DataSet(filenames=val_df[['L1path', 'L2path', 'L3path']],
                  labels=val_df['label'], covariate=val_df[COVARIATE], 
                  tile_weights=val_df['sample_weights'])
    tst = DataSet(filenames=tst_df[['L1path', 'L2path', 'L3path']], 
                  labels=tst_df['label'], covariate=tst_df[COVARIATE], 
                  tile_weights=tst_df['sample_weights'])


trn_ds = trn.create_dataset(batch_size=BATCH_SIZE, ds_epoch=None, augmentation=True)
val_ds = val.create_dataset(batch_size=BATCH_SIZE, ds_epoch=1)
tst_ds = tst.create_dataset(shuffle=False, batch_size=BATCH_SIZE, ds_epoch=1)


if MULTI_GPU:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = PANOPTES(base_model_name=BASE_MODEL, auxiliary=AUX, aux_weight=AUX_W, feature_pool=FEATURE_POOL, covariate=N_COV, dropout=DROPOUT, n_classes=NUM_CLASS)
        model.compile(loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(), 
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

else:
     model = PANOPTES(base_model_name=BASE_MODEL, auxiliary=AUX, aux_weight=AUX_W, feature_pool=FEATURE_POOL, covariate=N_COV, dropout=DROPOUT, n_classes=NUM_CLASS)
     model.compile(loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(), 
                   metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


model.train(trn_data=trn_ds, val_data=val_ds,
            steps=min(STEPS, MAX_STEPS),
            n_epoch=MAX_EPOCH,
            patience=PATIENCE,
            log_dir=OUT_DIR + '/log',
            model_dir=OUT_DIR + '/model')

model.print_attr()

print('Starting inference...')
tst_res = model.inference(tst_ds)

score_cols = ['Score_' + str(i) for i in range(tst_res[1].shape[1])]
print(score_cols)
tst_score = pd.DataFrame(tst_res[1], columns=score_cols)
tst_df = pd.concat([tst_df, tst_score], axis=1)

tst_df.to_csv(OUT_DIR + '/pred/tst_tile_pred.csv', index=False)

idx_cols = []
idx_cols.extend([AGG])
add_idx_cols = ['Patient_ID'] 
for element in add_idx_cols:
    if element not in idx_cols:
        idx_cols.append(element)    # 'Tumor_Normal', 'Patient_ID', 'Slide_ID', 'Tumor'

selected_cols = idx_cols.copy()
selected_cols.append('label')
selected_cols.extend(score_cols)
tst_df_slide = tst_df[selected_cols].groupby(idx_cols).agg('mean').reset_index() 

if NUM_CLASS == 2:
    fpr, tpr, thresholds = metrics.roc_curve(tst_df['label'], tst_df['Score_1'], pos_label=1)
    print('Tile level AUROC on test data: '  + str(metrics.auc(fpr, tpr)))

    print('Binary prediction. Metrics on positive scores.')
    fpr, tpr, thresholds = metrics.roc_curve(tst_df_slide['label'], tst_df_slide['Score_1'], pos_label=1)
    print('{} level AUROC on test data: '.format(AGG) + str(metrics.auc(fpr, tpr)))

else:
    print('Multi-class prediction. Per-class AUROC calculation.')
    for i in range(NUM_CLASS):
        fpr, tpr, thresholds = metrics.roc_curve(tst_df['label'], tst_df['Score_' + str(i)], pos_label=1)
        print('Tile level AUROC for level ' + str(i) + str(metrics.auc(fpr, tpr)))

        fpr, tpr, thresholds = metrics.roc_curve(tst_df_slide['label'], tst_df_slide['Score_' + str(i)], pos_label=1)
        print('{} level AUROC for level '.format(AGG) + str(i) + str(metrics.auc(fpr, tpr)))

print('Generating {} level tSNE...'.format(AGG))
activation_slide = pd.DataFrame(data=tst_res[0])    
activation_slide[AGG] = tst_df[AGG]
activation_slide = activation_slide.groupby([AGG]).agg('mean').reset_index()    # averaged activation for each slide

slide_embedding = pd.DataFrame(TSNE(n_components=2).fit_transform(activation_slide.drop(AGG, axis=1)),
                               columns=['tsne_0', 'tsne_1'])
slide_embedding[AGG] = activation_slide[AGG]

tst_df_slide = pd.merge(tst_df_slide, slide_embedding, how='left', on=AGG)
tst_df_slide.to_csv(OUT_DIR + '/pred/tst_slide_pred.csv', index=False)
print('{} level predictions and  tSNE embeddings saved.'.format(AGG))

print('Generating tile level tSNE...')
np.random.seed(SEED)
sample_idx = np.random.choice(tst_df.shape[0], MANIFOLD_SAMPLE)    # sample 20,000 tiles for TSNE
print('Activation shape: ' + str(tst_res[0].shape))
tst_sampled = tst_res[0][sample_idx, :]
tst_df_sampled = tst_df.copy()
tst_df_sampled = tst_df_sampled.iloc[sample_idx, :].reset_index(drop=True)

tile_embedding = pd.DataFrame(TSNE(n_components=2).fit_transform(tst_sampled),
                              columns=['tsne_0', 'tsne_1'])

tst_df_sampled = pd.concat([tst_df_sampled, tile_embedding], axis=1)

tst_df_sampled.to_csv(OUT_DIR + '/pred/tSNE_P_N.csv', index=False)
print('Tile level tSNE embeddings saved.' + ' {} tiles sampled.'.format(str(MANIFOLD_SAMPLE)))

