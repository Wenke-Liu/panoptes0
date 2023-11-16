import utils
import os
import math
import pandas as pd
import numpy as np 
from sklearn import metrics

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--setup', type=str, default='../results/STK11', help='Parent directory of the specific target.')
parser.add_argument('--exp', nargs='+', type=str, default=None, help='Experiment name. Default to all experiments.')
parser.add_argument('--variant', nargs='+', type=str, default=None, help='Model archtecture name. Default to all variants.')
parser.add_argument('--test', nargs='+', type=str, default='pred', help='Test subfolder.')
parser.add_argument('--num_class', type=int, default=2, help='Label classes.')
parser.add_argument('--agg_level', type=str, default='Patient_ID', help='Aggregation level of test inference.')  
parser.add_argument('--out_dir', type=str, default='../results/STK11', help='Output directory of aggregated metrics.')

args = parser.parse_args()
print('command line inputs:')
print(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))

"""
Global Variables
"""
SETUP = args.setup
EXP = args.exp
VARIANT = args.variant
TEST = args.test
NUM_CLASS = args.num_class
AGG = args.agg_level
OUT_DIR = args.out_dir

all_res = []

if EXP is None:
    exps = utils.get_subdirs(SETUP)    # Get all experiments for this setup
else:
    exps = EXP

for testset in TEST:
    for exp in exps:
        if VARIANT is None:
            variants = utils.get_subdirs(os.path.join(SETUP, exp))    # Get all model variants in the exp directory
        else:
            variants = VARIANT  

        for variant in variants:
            tile_pred = pd.read_csv(os.path.join(SETUP, exp, variant, testset, 'tst_tile_pred.csv'))
            score_cols = [col for col in tile_pred.columns if 'Score_' in col]
            idx_cols = ['Tumor_Normal', 'Tumor']    # 'Tumor_Normal', 'Patient_ID', 'Slide_ID', 'Tumor'
            idx_cols.extend([AGG])
            add_idx_cols = ['Patient_ID']    # always include Patient_ID
            for element in add_idx_cols:
                if element not in idx_cols:
                    idx_cols.append(element) 
               
            selected_cols = idx_cols.copy()
            selected_cols.append('label')
            selected_cols.extend(score_cols)
            agg_pred = tile_pred[selected_cols].groupby(idx_cols).agg('mean').reset_index()  

            if NUM_CLASS == 2:
                fpr, tpr, thresholds = metrics.roc_curve(tile_pred['label'], tile_pred['Score_1'], pos_label=1)
                tile_level = {'setup': os.path.basename(SETUP),
                          'testset': testset,
                          'exp': exp,
                          'variant': variant,
                          'level': 'tile',
                          'metric': 'AUROC',
                          'value': metrics.auc(fpr, tpr)}
                all_res.append(tile_level)
                fpr, tpr, thresholds = metrics.roc_curve(agg_pred['label'], agg_pred['Score_1'], pos_label=1)
                slide_level = {'setup': os.path.basename(SETUP),
                           'testset': testset,
                           'exp': exp,
                           'variant': variant,
                           'level': 'slide',
                           'metric': 'AUROC',
                           'value': metrics.auc(fpr, tpr)}
                all_res.append(slide_level)
    
            else:
                for i in range(NUM_CLASS):
                    fpr, tpr, thresholds = metrics.roc_curve(tile_pred['label'], tile_pred['Score_' + str(i)], pos_label=1)
                    tile_level = {'setup': os.path.basename(SETUP),
                              'testset': testset,
                              'exp': exp,
                              'variant': variant,
                              'class': i,
                              'level': 'tile',
                              'metric': 'AUROC',
                              'value': metrics.auc(fpr, tpr)}
                    all_res.append(tile_level)
                    fpr, tpr, thresholds = metrics.roc_curve(agg_pred['label'], agg_pred['Score_' + str(i)], pos_label=1)
                    slide_level = {'setup': os.path.basename(SETUP),
                               'testset': testset,
                               'exp': exp,
                               'variant': variant,
                               'class': i,
                               'level': 'slide',
                               'metric': 'AUROC',
                               'value': metrics.auc(fpr, tpr)}
                    all_res.append(slide_level)


res_df = pd.DataFrame(all_res)
res_df.to_csv(OUT_DIR + '/metrics.csv', index=False)        

sum_col = [col for col in res_df.columns if col not in ['value', 'exp']]

def std_err(x):
    return np.std(x)/math.sqrt(len(x))

sum_metrics = res_df.groupby(sum_col).agg(['mean', std_err]).reset_index()
sum_metrics.columns = ['setup', 'testset', 'variant', 'level', 'metric', 'mean', 'std_err']
sum_metrics.to_csv(OUT_DIR + '/metrics_summary.csv', index=False)






