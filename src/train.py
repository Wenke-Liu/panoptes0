import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from math import ceil,floor
from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
from data_input import DataSet
from model import PANOPTES
from sklearn.manifold import TSNE
from sklearn import metrics

#tf.config.experimental.set_memory_growth(True)

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--multi_gpu', type=str_to_bool, nargs='?', const=True, default=True,
                    help='Boolean. Whether to use multi-gpu training.')
parser.add_argument('--out_dir', type=str, default='./results/test', help='Parent output directory.')
parser.add_argument('--variant', type=str, default=None, help='Model variant abbreviation.')
parser.add_argument('--single_res', type=str, default=None, help='Column name of the single reolution path. e.g, L1path.')
parser.add_argument('--split', type=str_to_bool, nargs='?', const=True, default=False, 
                    help='Whether to split data or give trn, val and tst data tables.')
parser.add_argument('--split_ratio', type=str, default='0.7,0.15,0.15', help='Trn, val, tst ratio seperated by ,')
parser.add_argument('--tile_idx_dir', type=str, default='idx_files/pancan_imaging_all_tiles_index_TN.csv',
                    help='Path to tile index file.')
parser.add_argument('--label_df_dir', type=str, default='lab_files/pancan_1.1_ps_gene_drug_label.csv',
                    help='Path to label index file.')
parser.add_argument('--trn_df', type=str, default='./', help='Path to trn index file.')  
parser.add_argument('--val_df', type=str, default='./', help='Path to val index file.')  
parser.add_argument('--tst_df', type=str, default='./', help='Path to tst index file.')  
parser.add_argument('--lab_level', type=str, default='Patient_ID', help='Level of the labels. e.g, mutation on Patient_ID level, Tumor_Normal on Slide_ID level.')
parser.add_argument('--lab_col', type=str, default=None, help='Name of the label column in the label data table.')
parser.add_argument('--covariate', type=str, default=None, help='List of selected covarite columns, comma-delimited.')
parser.add_argument('--agg_level', type=str, default='Patient_ID', help='Aggregation level. Used for sub-sampling during training and aggregation during testing.')
parser.add_argument('--data_subsample', type=str, default=None, help='Wheather to subsample by agg_level. Two args seprated by comma should be given: threshold,ratio')
parser.add_argument('--val_sample', type=int, default=None, help='Number of validation examples used at the end of epoch.')
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
parser.add_argument('--legacy', type=str_to_bool, nargs='?', const=True, default=False, help='Whether to load data in legacy mode.') 

args = parser.parse_args()
print('command line inputs:')
print(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))

"""
Global variables, use argparse when run in cli
"""
MULTI_GPU = args.multi_gpu
VARIANT = args.variant
SINGLE_RES = args.single_res
COVARIATE = args.covariate
OUT_DIR = args.out_dir
SPLIT_RATIO = args.split_ratio
VAL_SAMPLE = args.val_sample
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
LEGACY = args.legacy

SPLIT_RATIO = tuple([float(x) for x in args.split_ratio.split(',')])

if VARIANT.startswith('I'):
    print('Single resolution model.')
    COVARIATE = None
    FEATURE_POOL = False
    if VARIANT == 'I5':
        BASE_MODEL = 'InceptionResNetV1'
    else:
        BASE_MODEL = 'InceptionResNetV2'
    if args.single_res is None:    # by default use level 1 when use single resolution models
        SINGLE_RES = 'L1path'

elif VARIANT.startswith('X', 'F'): 
    SINGLE_RES = None
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
   
print(f'Overridden by variant abbreviation if different:')    # use abbreviations
print(f'covariate: {str(COVARIATE)}')
print(f'base_model: {BASE_MODEL}')
print(f'Single resolution level: {SINGLE_RES}.')
print(f'feature_pool: {str(FEATURE_POOL)}')

try:
    os.makedirs(OUT_DIR + '/data')
    os.makedirs(OUT_DIR + '/pred')
except FileExistsError:
    pass

print('Experiment random seed: ' + str(SEED))
np.random.seed(SEED)
SPLIT_SEED, VAL_SEED = np.random.randint(low=0, high=1000000, size=2)
print('Data split seed: ' + str(SPLIT_SEED))
print('Validation sample seed: ' + str(VAL_SEED))


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

    print(f'Labels on {args.lab_level} level.')
    if args.lab_level == 'Patient_ID':
        data_idx = data_idx.merge(lab_df[['Patient_ID', 'Tumor', args.lab_col]], how='left')  # can change
    else:
        data_idx = data_idx.merge(lab_df[['Patient_ID', args.lab_level, 'Tumor', args.lab_col]], how='left')
    
    print('Renaming column {} into label column.'.format(args.lab_col))
    data_idx = data_idx.rename(columns={args.lab_col: 'label'})

    if COVARIATE is not None:
        print('Using covariates: ' + str(COVARIATE))
        
        data_idx = data_idx.merge(lab_df[COVARIATE + [args.lab_level]], how='left') 
        for cov in COVARIATE:
            data_idx[cov] = pd.to_numeric(data_idx[cov], errors='coerce')

    data_idx = data_idx.dropna()
    data_idx = data_idx.drop_duplicates()
    
    if args.data_subsample is not None:
        threshold, ratio = args.data_subsample.split(',')
        print(f'Subsample data by {AGG}.')
        data_idx = subsample_by_id(data_idx, id_col=AGG, threshold=int(threshold), ratio=float(ratio), seed=SPLIT_SEED)

    print('All data idx: {}'.format(str(data_idx.shape)))

    
    trn_id, val_id, tst_id = data_split(data_idx,
                                    split_ratio=SPLIT_RATIO,
                                    stratify='label',
                                    seed=SPLIT_SEED)
    
    trn_df = data_idx.loc[data_idx['Patient_ID'].isin(trn_id)].copy().reset_index(drop=True)    # always split by Patient_ID, regardless of lab_level
    
    val_df = data_idx.loc[data_idx['Patient_ID'].isin(val_id)].copy().reset_index(drop=True)
    if VAL_SAMPLE is not None:
        if VAL_SAMPLE > val_df.shape[0]:
            pass
        else:
            val_df = val_df.sample(n=VAL_SAMPLE, random_state=VAL_SEED).reset_index(drop=True)    # subsample the validation data
      
    tst_df = data_idx.loc[data_idx['Patient_ID'].isin(tst_id)].copy().reset_index(drop=True)
    

print('Applying training sample weights:')
trn_df = stratified_weights(trn_df, agg='Patient_ID')
trn_df = trn_df.sample(frac=1, random_state=SPLIT_SEED).reset_index(drop=True)
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=trn_df, fn='trn')


val_df['sample_weights'] = 1.0    # weighted val_df = stratified_weights(val_df, agg='Patient_ID')
val_df = val_df.sample(frac=1, random_state=SPLIT_SEED).reset_index(drop=True)
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=val_df, fn='val')

tst_df['sample_weights'] = 1.0  # unweighted
save_idx_df(out_dir=OUT_DIR + '/data', idx_df=tst_df, fn='tst')


print('Number of training examples: ' + str(trn_df.shape[0]))
MAX_TRN_STEPS = ceil(trn_df.shape[0]/BATCH_SIZE)
print('Maximum training steps per epoch: ' + str(MAX_TRN_STEPS))

print('Number of validation examples: ' + str(val_df.shape[0]))
VAL_STEPS = floor(min(VAL_SAMPLE, val_df.shape[0])/BATCH_SIZE)
print('Validation steps per epoch: ' + str(VAL_STEPS))

if SINGLE_RES is not None:
    im_paths = SINGLE_RES
else:
    im_paths = ['L1path', 'L2path', 'L3path']
if COVARIATE is None:
    trn = DataSet(filenames=trn_df[im_paths], labels=trn_df['label'], tile_weights=trn_df['sample_weights'])
    val = DataSet(filenames=val_df[im_paths], labels=val_df['label'], tile_weights=val_df['sample_weights'])
    tst = DataSet(filenames=tst_df[im_paths], labels=tst_df['label'], tile_weights=tst_df['sample_weights'])

else:
    
    trn = DataSet(filenames=trn_df[im_paths], 
                  labels=trn_df['label'], covariate=trn_df[COVARIATE], 
                  tile_weights=trn_df['sample_weights'], legacy=LEGACY)
    
    val = DataSet(filenames=val_df[im_paths],
                  labels=val_df['label'], covariate=val_df[COVARIATE], 
                  tile_weights=val_df['sample_weights'], legacy=LEGACY)
    
    tst = DataSet(filenames=tst_df[im_paths], 
                  labels=tst_df['label'], covariate=tst_df[COVARIATE], 
                  tile_weights=tst_df['sample_weights'], legacy=LEGACY)


trn_ds = trn.create_dataset(batch_size=BATCH_SIZE, ds_epoch=None, augmentation=True)
val_ds = val.create_dataset(batch_size=BATCH_SIZE, ds_epoch=None)
tst_ds = tst.create_dataset(shuffle=False, batch_size=BATCH_SIZE, ds_epoch=1)


if MULTI_GPU:
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
        model = PANOPTES(base_model_name=BASE_MODEL, single_res=SINGLE_RES, auxiliary=AUX, aux_weight=AUX_W, feature_pool=FEATURE_POOL, covariate=N_COV, dropout=DROPOUT, n_classes=NUM_CLASS)
        accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        model.compile(loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(), 
                      learning_rate=5e-5, metrics=[accuracy_metric])

else:
     model = PANOPTES(base_model_name=BASE_MODEL, single_res=SINGLE_RES, auxiliary=AUX, aux_weight=AUX_W, feature_pool=FEATURE_POOL, covariate=N_COV, dropout=DROPOUT, n_classes=NUM_CLASS)
     accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
     model.compile(loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(), 
                   learning_rate=5e-5, metrics=[accuracy_metric])


model.train(trn_data=trn_ds, val_data=val_ds,
            steps=min(STEPS, MAX_TRN_STEPS),
            n_epoch=MAX_EPOCH,
            patience=PATIENCE,
            val_steps=VAL_STEPS,
            log_dir=OUT_DIR + '/log',
            model_dir=OUT_DIR + '/model')

model.print_attr()

