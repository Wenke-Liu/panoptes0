import os
import argparse
import numpy as np
import tensorflow as tf
import pandas as pd
from math import ceil
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = str(pick_gpu_lowest_memory())
from tensorflow.keras.utils import to_categorical
from data_input import DataSet
from model import PANOPTES
from sklearn.manifold import TSNE
from sklearn import metrics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--multi_gpu', type=bool, default=True,
                    help='Boolean. Whether to use multi-gpu training.')
parser.add_argument('--out_dir', type=str, default='./results/test', help='Parent output directory.')

parser.add_argument('--split', type=bool, default=True, help='Whether to split data or give trn, val and tst data tables.')
parser.add_argument('--tile_idx_dir', type=str, default='idx_files/pancan_imaging_all_tiles_index_TN.csv',
                    help='Path to tile index file.')
parser.add_argument('--label_df_dir', type=str, default='lab_files/pancan_1.1_ps_gene_drug_label.csv',
                    help='Path to label index file.')
parser.add_argument('--trn_df', type=str, default='./', help='Path to trn index file.')  
parser.add_argument('--val_df', type=str, default='./', help='Path to val index file.')  
parser.add_argument('--tst_df', type=str, default='./', help='Path to tst index file.')  

parser.add_argument('--lab_col', type=str, default=None, help='Name of the label column in the label data table.')

parser.add_argument('--covariate', type=str, default=None, help='List of selected covarite columns.')

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

MULTI_GPU = args.multi_gpu
SPLIT = args.split
COVARIATE = args.covariate
OUT_DIR = args.out_dir
VAL_SAMPLE = args.val_sample
MANIFOLD_SAMPLE = args.manifold_sample
BATCH_SIZE = args.batch_size
STEPS = args.steps
MAX_EPOCH = args.max_epoch
PATIENCE = args.patience
SEED = args.seed

if COVARIATE is not None:
    COVARIATE = COVARIATE.split(',')   
    print('Using covariates: ' + str(COVARIATE))

