import pandas as pd
import numpy as np
import os
import pickle as pkl
import torch
import esm
from tqdm.auto import trange, tqdm
from datasets import load_dataset

def read_pkl(file_path):
    with open(file_path,'rb') as fr:
        return pkl.load(fr)

def save_pkl(file_path, val):
    fw = open(file_path, 'wb')
    pkl.dump(val, fw)
    fw.close()

# Load HuggingFace dataset instead of pickle files
print("Loading HuggingFace dataset...")
df = load_dataset("wanglab/cafa5", "cafa5_reasoning", split="train").to_pandas()

# Extract protein IDs and sequences
proteins = df['protein_id'].tolist()
x_seqs = df['sequence'].tolist()

print(f"Processing {len(proteins)} proteins...")

# Create output directory
os.makedirs('./processed_file/esm_emds', exist_ok=True)

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 10
steps = len(x_seqs)//batch_size + (1 if len(x_seqs) % batch_size != 0 else 0)

# Dictionary to store all embeddings
all_seq_reps = {}

for batch_idx in trange(steps):
    seq_reps = {}

    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(x_seqs))
    
    batch_seqs = x_seqs[start_idx:end_idx]
    batch_proteins = proteins[start_idx:end_idx]

    data = [(batch_idx*batch_size+idx, seq) for idx, seq in enumerate(batch_seqs)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    model = model.to(device)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[31], return_contacts=False)
    token_representations = results["representations"][31].to("cpu")
    
    for i, tokens_len in enumerate(batch_lens):
        all_seq_reps[batch_proteins[i]] = token_representations[i, 1 : tokens_len - 1].numpy()

# Save all embeddings as a single dictionary
print(f"Saving all {len(all_seq_reps)} embeddings...")
save_pkl('./processed_file/esm_emds/esm_all.pkl', all_seq_reps)

# Create protein map for backward compatibility (maps protein_id to batch_idx)
protein_map = {}
for i, pid in enumerate(proteins):
    protein_map[pid] = i // batch_size

save_pkl('./processed_file/protein_map.pkl', protein_map)

print("ESM processing complete!")
print(f"Total proteins processed: {len(all_seq_reps)}")
print(f"Files saved:")
print(f"  - Individual batches: ./processed_file/esm_emds/esm_part_*.pkl")
print(f"  - All embeddings: ./processed_file/esm_emds/esm_all.pkl")
print(f"  - Protein map: ./processed_file/protein_map.pkl")
