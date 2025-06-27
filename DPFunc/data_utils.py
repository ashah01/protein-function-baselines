import joblib
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm.auto import trange

import pickle as pkl
from scipy.sparse import csr_matrix

__all__ = ['get_go_list', 'get_mlb', 'get_pdb_data', 'get_inter_whole_data']


def get_go_list(pid_go_file, pid_list):
    pid_go = defaultdict(list)
    with open(pid_go_file) as fp:
        for line in fp:
            line_list=line.split()
            pid_go[(line_list)[0]].append(line_list[1])
    return [pid_go[pid_] for pid_ in pid_list]

def get_pdb_data(pdb_graph_file, pid_list):
    with open(pdb_graph_file, 'rb') as fr:
        pdb_graphs = pkl.load(fr)
    return [pdb_graphs.get(pid, None) for pid in pid_list]

def get_mlb(labels=None, **kwargs) -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer(sparse_output=False, **kwargs)
    mlb.fit(labels)
    return mlb

def get_inter_whole_data(interpro_list, save_file):
    if Path.exists(Path(save_file)):
        with open(save_file, 'rb') as fr:
            interpro_matrix = pkl.load(fr)
        return interpro_matrix
    
    rows = []
    cols = []
    data = []
    for i in trange(len(interpro_list)):
        vals_idx = interpro_list[i]
        val = [1]*len(vals_idx)
        
        rows += [i]*len(vals_idx)
        cols += vals_idx
        data += val
    
    col_nodes = 36062 # this value should be the same as the length of './data/inter_idx.pkl' 
    interpro_matrix = csr_matrix((data, (rows, cols)), shape=(len(interpro_list), col_nodes))
    with open(save_file, 'wb') as fw:
        pkl.dump(interpro_matrix, fw)
    
    return interpro_matrix

