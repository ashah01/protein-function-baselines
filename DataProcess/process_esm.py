import pandas as pd
import numpy as np
import os
import pickle as pkl
import torch
import esm
from tqdm.auto import trange, tqdm

def read_pkl(file_path):
    with open(file_path,'rb') as fr:
        return pkl.load(fr)

def save_pkl(file_path, val):
    fw = open(file_path, 'wb')
    pkl.dump(val, fw)
    fw.close()

pdb_seqs = read_pkl('./processed_file/pdb_seqs.pkl')
proteins, x_seqs = [], []
for k,v in pdb_seqs.items():
    proteins.append(k)
    x_seqs.append(v)
protein_map = {}

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device("cuda")

batch_size = 10
steps = len(x_seqs)//batch_size#+1
for batch_idx in trange(steps):
    seq_reps = {}

    data = [(batch_idx*batch_size+idx, seq) for idx, seq in enumerate(x_seqs[batch_idx*batch_size:(batch_idx+1)*batch_size])]
    prs = proteins[batch_idx*batch_size: (batch_idx+1)*batch_size]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    model = model.to(device)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[31], return_contacts=False)
    token_representations = results["representations"][31].to("cpu")
    
    for i, tokens_len in enumerate(batch_lens):
        seq_reps[prs[i]] = token_representations[i, 1 : tokens_len - 1].numpy()
    
    save_pkl('./processed_file/esm_emds/esm_part_{}.pkl'.format(batch_idx), seq_reps)

    for pr in prs:
        protein_map[pr] = batch_idx

save_pkl('./processed_file/protein_map.pkl', protein_map)
