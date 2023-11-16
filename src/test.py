from data_input import *
from model import PANOPTES
from utils import *
import pandas as pd
import numpy as np 
from sklearn import metrics
from sklearn.manifold import TSNE

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--multi_gpu', type=str_to_bool, nargs='?', const=True, default=True,
                    help='Boolean. Whether to use multi-gpu training.')
parser.add_argument('--variant', type=str, default=None, help='Model variant abbreviation.')
parser.add_argument('--single_res', type=str, default=None, help='Column name of the single reolution path. e.g, L1path.')
parser.add_argument('--out_dir', type=str, default='./results/test', help='Parent output directory.')

parser.add_argument('--tst_df', type=str, default=None, help='Path to tst index file.')  
parser.add_argument('--tile_idx_dir', type=str, default='idx_files/pancan_imaging_all_tiles_index_TN.csv',
                    help='Path to tile index file.')
parser.add_argument('--label_df_dir', type=str, default='lab_files/pancan_1.1_ps_gene_drug_label.csv',
                    help='Path to label index file.')
parser.add_argument('--lab_level', type=str, default='Patient_ID', 
                    help='Level of the labels. e.g, mutation on Patient_ID level, Tumor_Normal on Slide_ID level.')

parser.add_argument('--lab_col', type=str, default=None, help='Name of the label column in the label data table.')
parser.add_argument('--covariate', type=str, default=None, help='List of selected covarite columns, comma-delimited.')
parser.add_argument('--ids_per_chunk', type=int, default=5, help='Number of aggeregation ids in each chunk for inference. This is to prevent inference OOM.')
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
SINGLE_RES = args.single_res
COVARIATE = args.covariate
OUT_DIR = args.out_dir
IDS_PER_CHUNK = args.ids_per_chunk
MANIFOLD_SAMPLE = args.manifold_sample
BASE_MODEL = args.base_model
AUX = args.aux
NUM_CLASS = args.num_class
FEATURE_POOL = args.feature_pool
BATCH_SIZE = args.batch_size
AGG = args.agg_level
SEED = args.seed

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
    os.makedirs(OUT_DIR)

except FileExistsError:
    pass

if args.tst_df is not None:    # test dataframe given directly
    tst_df = pd.read_csv(args.tst_df)
    
else:    # combining tile indices and patient/slide level label dataframe
    print('Tile indices: ' + args.tile_idx_dir)
    tst_df = pd.read_csv(args.tile_idx_dir)
    tst_df = tst_df.dropna()
    print('Using label data: ' + args.label_df_dir)
    lab_df = pd.read_csv(args.label_df_dir)
    print(f'Labels on {args.lab_level} level.')
    if args.lab_level == 'Patient_ID':
        tst_df = tst_df.merge(lab_df[['Patient_ID', 'Tumor', args.lab_col]], how='left')  # can change
    else:
        tst_df = tst_df.merge(lab_df[['Patient_ID', args.lab_level, 'Tumor', args.lab_col]], how='left')  
    
    if COVARIATE is not None:
        print('Using covariates: ' + str(COVARIATE))
        
        tst_df = tst_df.merge(lab_df[COVARIATE + ['Patient_ID']], how='left') 
        for cov in COVARIATE:
            tst_df[cov] = pd.to_numeric(tst_df[cov], errors='coerce')

    tst_df = tst_df.dropna()

    print('All testing idx with labels: {}'.format(str(tst_df.shape)))

tst_df['sample_weights'] = 1  # unweighted

if args.lab_col is not None:
    print('Renaming column {} into label column.'.format(args.lab_col))
    tst_df = tst_df.rename(columns={args.lab_col: 'label'})

tst_df = tst_df.loc[~tst_df['label'].isna()]    # remove rows with missing labels
tst_df['label'] = pd.to_numeric(tst_df['label'], errors='coerce')    # convert label to numeric
tst_df = tst_df.reset_index(drop=True)    

if COVARIATE is not None:
    COVARIATE = COVARIATE.split(',')   # list of covariates 
    print('Using covariates: ' + str(COVARIATE))
    for col in COVARIATE:
        tst_df = tst_df.loc[~tst_df[col].isna()]    # remove rows with missing covariate
        tst_df = tst_df.reset_index(drop=True)    
        tst_df[col] = pd.to_numeric(tst_df[col], errors='coerce')
    N_COV = len(COVARIATE)
else:
    N_COV = None

idx_cols = []    # 'Tumor_Normal', 'Patient_ID', 'Slide_ID', 'Tumor'
idx_cols.extend([AGG])
add_idx_cols = ['Patient_ID']   # always include Patient_ID
for element in add_idx_cols:
    if element not in idx_cols:
        idx_cols.append(element) 
        
score_cols = ['Score_' + str(i) for i in range(NUM_CLASS)]
selected_cols = idx_cols.copy()
selected_cols.append('label')
selected_cols.extend(score_cols)   

model = PANOPTES(base_model_name=BASE_MODEL, single_res=SINGLE_RES, auxiliary=AUX, feature_pool=FEATURE_POOL, covariate=N_COV, n_classes=NUM_CLASS,
                 saved_model=args.saved_model_dir)
tst_df_ls = split_df_by_id(tst_df, n_id=3, agg=AGG)

print('Starting inference...', flush=True)
print(f'{len(tst_df_ls)} chunks with size {IDS_PER_CHUNK} each.', flush=True)

if SINGLE_RES is not None:
    im_paths = SINGLE_RES
else:
    im_paths = ['L1path', 'L2path', 'L3path']

tst_res_ls = []

for i in range(len(tst_df_ls)):
    if COVARIATE is not None:
        tst_chunk = DataSet(filenames=tst_df_ls[i][im_paths], 
                  labels=tst_df_ls[i]['label'], covariate=tst_df_ls[i][COVARIATE], 
                  tile_weights=tst_df_ls[i]['sample_weights'], legacy=args.legacy)
    else:
        tst_chunk = DataSet(filenames=tst_df_ls[i][im_paths],
              labels=tst_df_ls[i]['label'], 
              tile_weights=tst_df_ls[i]['sample_weights'], legacy=args.legacy)
    
    tst_chunk_ds = tst_chunk.create_dataset(shuffle=False, batch_size=BATCH_SIZE, ds_epoch=1)
    chunk_latent, chunk_score = model.inference(tst_chunk_ds)
    print(f'Chunk {i} finished.', flush=True)

    chunk_latent = pd.DataFrame(chunk_latent)    
    chunk_latent[AGG] = tst_df_ls[i][AGG]
    chunk_latent = chunk_latent.groupby([AGG]).agg('mean').reset_index()    # averaged latents for each aggregation level (usually slide)

    chunk_score = pd.DataFrame(chunk_score, columns=score_cols)    # tile level scores
    chunk_score = pd.concat([tst_df_ls[i], chunk_score], axis=1)
    tst_res_chunk = [chunk_latent, chunk_score]
    tst_res_ls.append(tst_res_chunk)


if len(tst_res_ls) == 1:
    agg_latents = tst_res_ls[0][0]
    tile_scores = tst_res_ls[0][1]

else:
    latents = [item[0] for item in tst_res_ls]
    scores = [item[1] for item in tst_res_ls]
    agg_latents = pd.concat(latents, axis=0)    # latents aggregated first 
    tile_scores = pd.concat(scores, axis=0)

tile_scores.to_csv(OUT_DIR + '/tst_tile_pred.csv', index=False)
print('Tile level predictions saved.')

agg_scores = tile_scores[selected_cols].groupby(idx_cols).agg('mean').reset_index()  

if NUM_CLASS == 2:
    print('Binary prediction. Metrics on positive scores.')

    fpr, tpr, thresholds = metrics.roc_curve(tile_scores['label'], tile_scores['Score_1'], pos_label=1)
    print('Tile level AUROC on test data: '  + str(metrics.auc(fpr, tpr)))

    fpr, tpr, thresholds = metrics.roc_curve(agg_scores['label'], agg_scores['Score_1'], pos_label=1)
    print('{} level AUROC on test data: '.format(AGG) + str(metrics.auc(fpr, tpr)))

else:
    print('Multi-class prediction. Per-class AUROC calculation.')
    for i in range(NUM_CLASS):
        fpr, tpr, thresholds = metrics.roc_curve(tile_scores['label'], tile_scores['Score_' + str(i)], pos_label=1)
        print('Tile level AUROC for level ' + str(i) + str(metrics.auc(fpr, tpr)))
        
        fpr, tpr, thresholds = metrics.roc_curve(agg_scores['label'], agg_scores['Score_' + str(i)], pos_label=1)
        print('{} level AUROC for level '.format(AGG) + str(i) + str(metrics.auc(fpr, tpr)))


print('Generating {} level tSNE...'.format(AGG))

agg_embedding = pd.DataFrame(TSNE(n_components=2).fit_transform(agg_latents.drop(AGG, axis=1)),
                               columns=['tsne_0', 'tsne_1'])
agg_embedding[AGG] = agg_latents[AGG].values

agg_scores = pd.merge(agg_scores, agg_embedding, how='left', on=AGG)
agg_scores.to_csv(OUT_DIR + '/tst_slide_pred.csv', index=False)
print('{} level predictions and  tSNE embeddings saved.'.format(AGG))

print('Generating tile level tSNE...')

MANIFOLD_SAMPLE = min(MANIFOLD_SAMPLE, tst_df.shape[0])
tst_sampled_df = tst_df.sample(n=MANIFOLD_SAMPLE, random_state=SEED)    # sample MANIFOLD_SAMPLE tiles for TSNE

if COVARIATE is not None:
    tst_sampled = DataSet(filenames=tst_sampled_df[['L1path', 'L2path', 'L3path']], 
                        labels=tst_sampled_df['label'], covariate=tst_sampled_df[COVARIATE], 
                        tile_weights=tst_sampled_df['sample_weights'], legacy=args.legacy)
else:
    tst_sampled = DataSet(filenames=tst_sampled_df[['L1path', 'L2path', 'L3path']],
                        labels=tst_sampled_df['label'], 
                        tile_weights=tst_sampled_df['sample_weights'], legacy=args.legacy)
    
tst_sampled_ds = tst_sampled.create_dataset(shuffle=False, batch_size=BATCH_SIZE, ds_epoch=1)

sampled_latent, sampled_score = model.inference(tst_sampled_ds)
print(f'Activation shape: {str(sampled_latent.shape)}.')

sampled_score = pd.DataFrame(sampled_score, columns=score_cols)
sampled_score = pd.concat([tst_sampled_df, sampled_score], axis=1)
sampled_embedding = pd.DataFrame(TSNE(n_components=2).fit_transform(sampled_latent),
                              columns=['tsne_0', 'tsne_1'])

sampled_score = pd.concat([sampled_score, sampled_embedding], axis=1)

sampled_score.to_csv(OUT_DIR + '/tSNE_P_N.csv', index=False)
print('Tile level tSNE embeddings saved.' + ' {} tiles sampled.'.format(str(MANIFOLD_SAMPLE)))

