import pandas as pd
import numpy as np
import os
import pickle as pkl
import torch
import esm
from tqdm.auto import trange, tqdm
from load_cafa5 import load_cafa5_dataset

def read_pkl(file_path):
    with open(file_path,'rb') as fr:
        return pkl.load(fr)

def save_pkl(file_path, val):
    fw = open(file_path, 'wb')
    pkl.dump(val, fw)
    fw.close()

# Load HuggingFace dataset instead of pickle files
print("Loading HuggingFace dataset...")
train_dataset, val_dataset, test_dataset = load_cafa5_dataset(
    dataset="wanglab/cafa5",
    dataset_name="cafa5_reasoning",
    dataset_subset=None,
    max_length=2048,
    val_split_ratio=0.1,
    seed=23,
    return_as_chat_template=False,
    cache_dir='./cafa5',
    structure_dir='./cafa5/extracted' # '/Users/arnavshah/Code/DPFunc/cafa5/extracted'
)

proteins = train_dataset['protein_id']
x_seqs = train_dataset['sequence']

print(f"Processing {len(proteins)} proteins...")

# Create output directory
os.makedirs('/large_storage/goodarzilab/ashah/esm_emds_train', exist_ok=True)

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 10
steps = len(x_seqs)//batch_size + (1 if len(x_seqs) % batch_size != 0 else 0)
protein_map = {}
seq_reps = {}

for batch_idx in trange(steps):
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

    if (batch_idx + 1) % 10 == 0:
        save_pkl('/large_storage/goodarzilab/ashah/esm_emds_train/esm_part_{}.pkl'.format(batch_idx // 10), seq_reps)
        seq_reps = {}

    for pr in prs:
        protein_map[pr] = batch_idx // 10

save_pkl('/large_storage/goodarzilab/ashah/esm_train_protein_map.pkl', protein_map)

# Create protein map for backward compatibility (maps protein_id to batch_idx)
# protein_map = {}
# for i, pid in enumerate(proteins):
#     protein_map[pid] = i // batch_size

# save_pkl('./processed_file/protein_map.pkl', protein_map)
