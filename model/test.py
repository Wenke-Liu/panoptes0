from data_input import *
from model import *
from utils import *
import pandas as pd
import numpy as np 
from sklearn import metrics
from sklearn.manifold import TSNE

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--multi_gpu', type=str_to_bool, nargs='?', const=True, default=False,
                    help='Boolean. Whether to use multi-gpu training.')
parser.add_argument('--variant', type=str, default=None, help='Model variant abbreviation.')
parser.add_argument('--out_dir', type=str, default='./results/test', help='Parent output directory.')

parser.add_argument('--tst_df', type=str, default='./', help='Path to tst index file.')  

parser.add_argument('--lab_col', type=str, default=None, help='Name of the label column in the label data table.')
parser.add_argument('--covariate', type=str, default=None, help='List of selected covarite columns, comma-delimited.')
parser.add_argument('--manifold_sample', type=int, default=20000, help='Number of test examples used to estimate manifold.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')

parser.add_argument('--base_model', type=str, default='InceptionResNetV1', help='Name of the branch base model.')  
parser.add_argument('--aux', type=str_to_bool, nargs='?', const=True, default=False, help='Whether use auxiliary outputs for training.') 
parser.add_argument('--feature_pool', type=str_to_bool, nargs='?', const=True, default=False, help='Whether include feature pooling')  
parser.add_argument('--num_class', type=int, default=2, help='Label classes.')
parser.add_argument('--batch_size', type=int, default=16, help='Training and testing batch size.')
parser.add_argument('--saved_model_dir', type=str, default='./results/test/model/', 
                    help='Directory to saved model checkpoints.')  
parser.add_argument('--agg_level', type=str, default='Patient_ID', help='Aggregation level of test inference.')  
parser.add_argument('--legacy', type=str_to_bool, nargs='?', const=True, default=False, help='Whether to load data in legacy mode.') 

args = parser.parse_args()
print('command line inputs:')
print(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))

"""
Global Variables
"""
MULTI_GPU = args.multi_gpu
VARIANT = args.variant
COVARIATE = args.covariate
OUT_DIR = args.out_dir
MANIFOLD_SAMPLE = args.manifold_sample
BASE_MODEL = args.base_model
AUX = args.aux
NUM_CLASS = args.num_class
FEATURE_POOL = args.feature_pool
BATCH_SIZE = args.batch_size
AGG = args.agg_level
SEED = args.seed

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
    
    print('Overridden by abbreviation if different:\n')
    print('base_model: {}'.format(BASE_MODEL))
    print('feature_pool: {}'.format(str(FEATURE_POOL)))
    
try:
    os.makedirs(OUT_DIR)
    os.makedirs(OUT_DIR)
except FileExistsError:
    pass

tst_df = pd.read_csv(args.tst_df)
tst_df['sample_weights'] = 1  # unweighted

if args.lab_col is not None:
    print('Renaming column {} into label column.'.format(args.lab_col))
    tst_df = tst_df.rename(columns={args.lab_col: 'label'})

tst_df = tst_df.loc[~tst_df['label'].isna()]    # remove rows with missing labels
tst_df = tst_df.reset_index(drop=True)    

if COVARIATE is not None:
    COVARIATE = COVARIATE.split(',')   # list of covariates 
    print('Using covariates: ' + str(COVARIATE))
    for col in COVARIATE:
        tst_df = tst_df.loc[~tst_df[col].isna()]    # remove rows with missing covariate
        tst_df = tst_df.reset_index(drop=True)    
    N_COV = len(COVARIATE)
    tst = DataSet(filenames=tst_df[['L1path', 'L2path', 'L3path']], 
                  labels=tst_df['label'], covariate=tst_df[COVARIATE], 
                  tile_weights=tst_df['sample_weights'], id_level=3, legacy=args.legacy)
 
else:
    N_COV = None
    tst = DataSet(filenames=tst_df[['L1path', 'L2path', 'L3path']],
              labels=tst_df['label'], 
              tile_weights=tst_df['sample_weights'], id_level=3, legacy=args.legacy)

tst_ds = tst.create_dataset(shuffle=False, batch_size=BATCH_SIZE, ds_epoch=1)

model = PANOPTES(base_model_name=BASE_MODEL, auxiliary=AUX, feature_pool=FEATURE_POOL, covariate=N_COV, n_classes=NUM_CLASS,
                 saved_model=args.saved_model_dir)
print('Starting inference...')
tst_res = model.inference(tst_ds)

score_cols = ['Score_' + str(i) for i in range(tst_res[1].shape[1])]

tst_score = pd.DataFrame(tst_res[1], columns=score_cols)
tst_df = pd.concat([tst_df, tst_score], axis=1)

tst_df.to_csv(OUT_DIR + '/tst_tile_pred.csv', index=False)
print('Tile level predictions saved.')

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
    print('Binary prediction. Metrics on positive scores.')
    fpr, tpr, thresholds = metrics.roc_curve(tst_df_slide['label'], tst_df_slide['Score_1'], pos_label=1)
    print('{} level AUROC on test data: '.format(AGG) + str(metrics.auc(fpr, tpr)))

else:
    print('Multi-class prediction. Per-class AUROC calculation.')
    for i in range(NUM_CLASS):
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
tst_df_slide.to_csv(OUT_DIR + '/tst_slide_pred.csv', index=False)
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

tst_df_sampled.to_csv(OUT_DIR + '/tSNE_P_N.csv', index=False)
print('Tile level tSNE embeddings saved.' + ' {} tiles sampled.'.format(str(MANIFOLD_SAMPLE)))

