# import torch
# import dgl
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from datasets import load_dataset
from Bio.PDB.MMCIFParser import FastMMCIFParser
from Bio.PDB.PDBParser import PDBParser
import fsspec
import re
import math
import gzip
import pickle as pkl
from scipy.sparse import csr_matrix
from tqdm import trange

df = load_dataset("wanglab/cafa5", "cafa5_reasoning", split="train").to_pandas()

def read_pkl(file_path):
    with open(file_path,'rb') as fr:
        return pkl.load(fr)

def save_pkl(file_path, val):
    fw = open(file_path, 'wb')
    pkl.dump(val, fw)
    fw.close()


def extract_sequence_and_ca_coords(pdb_file, chain_id=None, af=True):
    parser = FastMMCIFParser(QUIET=True) if af else PDBParser(QUIET=True)
    if pdb_file.endswith(".gz"):
        with gzip.open(pdb_file, "rt") as gz_file:
            temp_file = pdb_file.replace(".gz", "_temp")
            try:
                with open(temp_file, "w") as temp:
                    temp.write(gz_file.read())

                structure = parser.get_structure("protein", temp_file)
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    else:
        structure = parser.get_structure("protein", pdb_file)

    results = {}

    for model in structure:
        for chain in model:
            if chain_id is None or chain.id == chain_id:
                ca_coords = []

                for residue in chain:
                    if residue.id[0] == " ":
                        try:
                            if "CA" in residue:
                                ca_atom = residue["CA"]
                                coord = ca_atom.get_coord()
                                ca_coords.append(
                                    (float(coord[0]), float(coord[1]), float(coord[2]))
                                )
                            else:
                                print(
                                    f"Warning: No CA atom found in residue {residue.resname}{residue.id[1]} of chain {chain.id}"
                                )
                                ca_coords.append(None)

                        except KeyError:
                            print(f"Non natural residue in {pdb_file}")

                results[chain.id] = {"ca_coords": ca_coords}

    return results[chain_id]["ca_coords"] if chain_id in results else []


fs = fsspec.filesystem("file")

# we don't know the shard so we'll just regex match based on structure in hf
shard_paths = fs.glob(
    "/Users/arnavshah/Code/DPFunc/cafa5/structures/*/*/*"
)  # Lists all tar.gz shard paths
af_shard_index = [
    re.search(
        r"AF-(.*?)-F1-",
        sorted(
            fs.glob(
                f"/Users/arnavshah/Code/DPFunc/cafa5/structures/af_shards/shard_{i}/*"
            )
        )[-1].split("/")[-1],
    ).group(1)
    for i in range(35)
]
pdb_shard_index = [
    sorted(
        fs.glob(f"/Users/arnavshah/Code/DPFunc/cafa5/structures/pdb_shards/shard_{i}/*")
    )[-1]
    .split("/")[-1]
    .split(".")[0]
    for i in range(5)
]


def find_protein_shard(entry: str, af: bool = True) -> int:
    code = re.search(r"AF-(.*?)-F1-", entry).group(1) if af else entry.split(".")[0]
    shard_index = af_shard_index if af else pdb_shard_index
    length = 35 if af else 5

    for i in range(length):
        if code <= shard_index[i]:
            return i


pdb_points_info = {}
pdb_seq_info = {}
unseen_proteins = set()

for index, row in tqdm(df.iloc[:50].iterrows()):
    uni_id, sequence, struct_entry = (
        row["protein_id"],
        row["sequence"],
        row["structure_path"],
    )
    if struct_entry is None:  # will be roughly 5%
        unseen_proteins.add(uni_id)
        continue

    database, entry = struct_entry.split("/")
    af = database == "af_db"
    assert database in ["af_db", "pdb_files"]
    pdb_file = f"../cafa5/structures/{'af_shards' if af else 'pdb_shards'}/shard_{find_protein_shard(entry, af)}/{entry}"

    if not os.path.exists(pdb_file):  # should never be triggered (@Purav)
        print(f"GUARD REACHED {struct_entry}. PDB file: {pdb_file}")
        unseen_proteins.add(uni_id)
        continue

    coords_list = extract_sequence_and_ca_coords(pdb_file, "A", af)

    if coords_list:  # guard
        valid_coords = [coord for coord in coords_list if coord is not None]
        pdb_points_info[uni_id] = valid_coords
        pdb_seq_info[uni_id] = sequence

# save_pkl('./processed_file/pdb_points.pkl', pdb_points_info)
# save_pkl('./processed_file/pdb_seqs.pkl', pdb_seq_info)
# save_pkl('./processed_file/unseen_proteins.pkl', unseen_proteins)


# pdb_seqs = pdb_seq_info


# def get_dis(point1, point2):
#     dis_x = point1[0] - point2[0]
#     dis_y = point1[1] - point2[1]
#     dis_z = point1[2] - point2[2]
#     return math.sqrt(dis_x * dis_x + dis_y * dis_y + dis_z * dis_z)


# def process_input_pdb_file(
#     tag, part, pid_list, pdb_points_info, pdb_seqs, thresholds=12
# ):
#     protein_map = read_pkl("./processed_file/protein_map.pkl")
#     pdb_graphs = []
#     p_cnt = 0
#     file_idx = 0
#     for pid in tqdm(pid_list):
#         p_cnt += 1
#         points = pdb_points_info[pid]

#         u_list = []
#         v_list = []
#         dis_list = []
#         for uid, amino_1 in enumerate(points):
#             for vid, amino_2 in enumerate(points):
#                 if uid == vid:
#                     continue
#                 dist = get_dis(amino_1, amino_2)
#                 if dist <= thresholds:
#                     u_list.append(uid)
#                     v_list.append(vid)
#                     dis_list.append(dist)
#         u_list, v_list = torch.tensor(u_list), torch.tensor(v_list)
#         dis_list = torch.tensor(dis_list)

#         graph = dgl.graph((u_list, v_list), num_nodes=len(points))
#         graph.edata["dis"] = dis_list

#         # graph node feature - esm
#         esm_file_idx = protein_map[pid]
#         esm_features = read_pkl(
#             f"./processed_file/esm_emds/esm_part_{esm_file_idx}.pkl"
#         )
#         node_features = esm_features[pid]
#         assert node_features.shape[0] == graph.num_nodes()
#         graph.ndata["x"] = torch.from_numpy(node_features)
#         pdb_graphs.append(graph)

#         if p_cnt % 5000 == 0:
#             save_pkl(
#                 "./processed_file/graph_features/{}_{}_whole_pdb_part{}.pkl".format(
#                     tag, part, file_idx
#                 ),
#                 pdb_graphs,
#             )
#             p_cnt = 0
#             file_idx += 1
#             pdb_graphs = []
#     if len(pdb_graphs) > 0:
#         save_pkl(
#             "./processed_file/graph_features/{}_{}_whole_pdb_part{}.pkl".format(
#                 tag, part, file_idx
#             ),
#             pdb_graphs,
#         )
#     return file_idx


# for tag in tags:
#     if tag == "mf":
#         continue
#     for tp in types:
#         pid_list = read_pkl(f"./processed_file/{tag}_{tp}_used_pid_list.pkl")
#         max_cnt = process_input_pdb_file(tag, tp, pid_list, pdb_points_info, pdb_seqs)
#         if tp == "train":
#             print(f"{tag}-{tp}-train_file_count-{max_cnt}")

interpro_list = read_pkl("interpro_list.pkl")
inter_idx = {}
for idx, ipr in enumerate(interpro_list):
    inter_idx[ipr] = idx

# NOTE: I don't like this interpro approach because it's not order agnostic. If i want to fetch a different protein id (maybe some structure
# NOTE: file is missing, or I've shuffled, or anything else) I'm completely screwed.

# TODO: We'll need to use the cafa5 huggingface dataset in the training loop anyway to query the correct graph file (the only information we should actually be precomputing and saving on disk).
# TODO: Let's change the training loop to only take advantage of that precomputation and rely on huggingface for everything else.

inter_matrices = []
for index, row in df.head(50).iterrows():
    inters = row["interpro_ids"]
    if inters is None:
        continue
    inter_matrix = np.zeros(len(interpro_list))
    for it in inters:
        inter_matrix[inter_idx[it]] += 1
    inter_matrices.append(inter_matrix)


rows = []
cols = []
data = []
for i in trange(len(inter_matrices)):
    inter_matrix = inter_matrices[i]
    vals_idx = np.argwhere(inter_matrix>0).reshape(-1)
    val = inter_matrix[vals_idx]
        
    rows += [i]*len(vals_idx)
    cols += vals_idx.tolist()
    data += val.tolist()

col_nodes = 36062 # this value should be the same as the length of './data/inter_idx.pkl' 
interpro_matrix = csr_matrix((data, (rows, cols)), shape=(len(inter_matrices), col_nodes))
import IPython; IPython.embed()
