import os
import math
import pandas as pd
import sklearn.utils as sku
import numpy as np
import re
import subprocess, re


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def get_subdirs(root_dir):
    subdirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    return subdirs

def split_dataframe(df, chunk_size):
    """Split a DataFrame into chunks of a specified size."""
    chunks = []
    num_chunks = math.ceil(len(df) / chunk_size)

    for i in range(num_chunks):
        chunks.append(df[i*chunk_size:(i+1)*chunk_size])

    return chunks

def split_df_by_id(df, agg, n_id):
    """Split a DataFrame into chunks by n_id of aggregation id."""
    chunks = []
    ids = df[agg].unique()
    n_chunks = math.ceil(len(ids) / n_id)
    
    for i in range(n_chunks):
        chunk_ids = ids[i*n_id : (i+1)*n_id]
        chunk = df.loc[df[agg].isin(chunk_ids)].copy().reset_index(drop=True) 
        chunks.append(chunk)

    return chunks

def subsample_by_id(df, id_col='Slide_ID', threshold=500, ratio=0.1, seed=42):
    """Subsample tiles from the same id """
    id_chunks = split_df_by_id(df, agg=id_col, n_id=1)
    id_counts = [chunk.shape[0] for chunk in id_chunks]
    id_sub = []
    for count in id_counts:
        if count <= threshold:
            id_sub.append(count)
        else:
            id_sub.append(math.floor((count -threshold) * ratio) + threshold)
    np.random.seed(seed)
    seeds = np.random.randint(low=0, high=1000000, size=len(id_counts))
    sub_chunks = []
    for i in range(len(id_counts)):
        chunk = id_chunks[i]
        sub_chunks.append(chunk.sample(n=id_sub[i], random_state=seeds[i]).reset_index(drop=True))
    
    return pd.concat(sub_chunks)    


# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23
def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")

def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse "+line
        result.append(int(m.group("gpu_id")))
    return result

def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result

def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu

def data_split(df,
               id='Patient_ID',
               stratify='label',
               split_ratio=(0.8, 0.1, 0.1),
               collapse=True,
               seed=42):

    print('Using {} as id column.'.format(str(id)))
    print('Split ratio: ' + str(split_ratio))
    levels = df[stratify].unique()
    levels.sort()
    trn = []
    val = []
    tst = []
    trn_sizes = []
    val_sizes = []
    tst_sizes = []
    np.random.seed(seed)
    seeds = np.random.randint(low=0, high=1000000, size=len(levels))

    for i, level in enumerate(levels):    # stratified splits
        ids = df.loc[df[stratify] == level][id].unique()
        print('{} unique ids in {}'.format(str(len(ids)), str(level)))
        
        val_size = math.floor(len(ids)*split_ratio[1])
        tst_size = math.floor(len(ids)*split_ratio[2])
        trn_size = len(ids) - (val_size + tst_size)

        trn_sizes.append(trn_size)
        val_sizes.append(val_size)
        tst_sizes.append(tst_size)

        np.random.seed(seeds[i])    
        np.random.shuffle(ids)
        
        trn.append(ids[:trn_size])
        val.append(ids[trn_size: (trn_size + val_size)])
        tst.append(ids[(trn_size + val_size):])
        
    print('Training samples: ' + str(trn_sizes))
    print('Validation samples: ' + str(val_sizes))
    print('Testing samples: ' + str(tst_sizes))

    if collapse:
        print('Collapsing ids in each split.')
        trn = np.concatenate(trn)
        val = np.concatenate(val)
        tst = np.concatenate(tst)

    return trn, val, tst


def paired_tile_ids_in(patient, slide, tumor, root_dir):    # pancan version
    dira = os.path.isdir(root_dir + '/level1')
    dirb = os.path.isdir(root_dir + '/level2')
    dirc = os.path.isdir(root_dir + '/level3')
    if dira and dirb and dirc:
        fac = 1000
        ids = []
        for level in range(1, 4):
            dirr = root_dir + '/level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id:
                    x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                    y = int(float(re.split('.p', id.split('y-', 1)[1])[0]) / fac)
                    ids.append([patient, slide, tumor, level, dirr + '/' + id, x, y])
        ids = pd.DataFrame(ids, columns=['Patient_ID', 'Slide_ID', 'Tumor', 'level', 'path', 'x', 'y'])
        idsa = ids.loc[ids['level'] == 1]
        idsa = idsa.drop(columns=['level'])
        idsa = idsa.rename(index=str, columns={"path": "L1path"})
        idsb = ids.loc[ids['level'] == 2]
        idsb = idsb.drop(columns=['Patient_ID', 'Slide_ID', 'Tumor', 'level'])
        idsb = idsb.rename(index=str, columns={"path": "L2path"})
        idsc = ids.loc[ids['level'] == 3]
        idsc = idsc.drop(columns=['Patient_ID', 'Slide_ID', 'Tumor', 'level'])
        idsc = idsc.rename(index=str, columns={"path": "L3path"})
        idsa = pd.merge(idsa, idsb, on=['x', 'y'], how='left', validate="many_to_many")
        idsa['x'] = idsa['x'] - (idsa['x'] % 2)
        idsa['y'] = idsa['y'] - (idsa['y'] % 2)
        idsa = pd.merge(idsa, idsc, on=['x', 'y'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y'])
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
    else:
        print('Pass: ', root_dir)  # exclude dir doesn't have all three levels
        idsa = pd.DataFrame(columns=['Patient_ID', 'Slide_ID', 'Tumor', 'L1path', 'L2path', 'L3path'])

    return idsa

def paired_tile_ids_in2(patient, tumor, root_dir):    # UCEC version
    dira = os.path.isdir(root_dir + '/level1')
    dirb = os.path.isdir(root_dir + '/level2')
    dirc = os.path.isdir(root_dir + '/level3')
    if dira and dirb and dirc:
        if "TCGA" in root_dir:
            fac = 2000
        else:
            fac = 1000
        ids = []
        for level in range(1, 4):
            dirr = root_dir + '/level{}'.format(str(level))
            for id in os.listdir(dirr):
                if '.png' in id:
                    x = int(float(id.split('x-', 1)[1].split('-', 1)[0]) / fac)
                    y = int(float(re.split('_', id.split('y-', 1)[1])[0]) / fac)
                    try:
                        slide = re.split('.p', re.split('_', id.split('y-', 1)[1])[1])[0]
                    except IndexError:
                        slide = '00'
                    ids.append([patient, slide, tumor, level, dirr + '/' + id, x, y])
                else:
                    print('Skipping ID:', id)
        ids = pd.DataFrame(ids, columns=['Patient_ID', 'Slide_ID', 'Tumor', 'level', 'path', 'x', 'y'])
        idsa = ids.loc[ids['level'] == 1]
        idsa = idsa.drop(columns=['level'])
        idsa = idsa.rename(index=str, columns={"path": "L1path"})
        idsb = ids.loc[ids['level'] == 2]
        idsb = idsb.drop(columns=['Patient_ID', 'Tumor', 'level'])
        idsb = idsb.rename(index=str, columns={"path": "L2path"})
        idsc = ids.loc[ids['level'] == 3]
        idsc = idsc.drop(columns=['Patient_ID', 'Tumor', 'level'])
        idsc = idsc.rename(index=str, columns={"path": "L3path"})
        idsa = pd.merge(idsa, idsb, on=['Slide_ID', 'x', 'y'], how='left', validate="many_to_many")
        idsa['x'] = idsa['x'] - (idsa['x'] % 2)
        idsa['y'] = idsa['y'] - (idsa['y'] % 2)
        idsa = pd.merge(idsa, idsc, on=['Slide_ID', 'x', 'y'], how='left', validate="many_to_many")
        idsa = idsa.drop(columns=['x', 'y'])
        idsa = idsa.dropna()
        idsa = sku.shuffle(idsa)
    else:
        print('Pass: ', root_dir)  # exclude dir doesn't have all three levels
        idsa = pd.DataFrame(columns=['Patient_ID', 'Slide_ID', 'Tumor', 'L1path', 'L2path', 'L3path'])

    return idsa


def save_idx_df(out_dir, idx_df, fn):
    if not 'Slide_ID' in idx_df.columns:
        idx_df['Slide_ID'] = idx_df['Patient_ID']
        
    idx_df.to_csv(out_dir + '/' + fn + '_tile_idx.csv', index=False)
    slide_idx_df = idx_df.copy()
    slide_idx_df = slide_idx_df[['Patient_ID', 'Slide_ID', 'Tumor', 'label']].drop_duplicates()

    slide_idx_df.to_csv(out_dir + '/' + fn + '_slide_idx.csv', index=False)


def class_weight(label, power=1./3):
    freq = np.bincount(label)
    total = np.sum(freq)
    class_weight = ((1/freq)*(total/len(freq))) ** power
    class_weight = dict(zip(list(range(0,len(freq))),class_weight))
    
    return class_weight
    

def stratified_weights(df, stratify='Tumor', agg=None, weighted='label', power=1./3, verbose=True):
    all_weights = {}
    df_w = df.copy()[[stratify, weighted]]
    if agg is not None:    # add aggregation factor
        df_w[agg] = df[agg]
    
    print('Power of weights: ' + str(power))
    for level in df_w[stratify].unique():
        df_w_level = df_w.loc[df_w[stratify] == level]
        if len(df_w_level[weighted].value_counts())==1:    # only have one outcome
            # weights = {0:0,1:0}
            weights = dict(zip(list(range(0,len(df_w['label'].unique()))),[0]*len(df_w['label'].unique())))
        else:
            if agg is not None:    # aggregate and then calculate weights
                raw_label = df_w_level.groupby(agg)[weighted].agg('mean')
            else:    # weights on instance level
                raw_label = df_w_level[weighted]
            weights = class_weight(raw_label, power=power)
        
        all_weights[level] = weights
        
    if verbose:
        print(pd.DataFrame(all_weights))
        
    df['sample_weights'] = [all_weights[x][y] for x, y in zip(df[stratify], df[weighted])]
        
    return df


