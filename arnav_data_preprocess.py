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

# Load HuggingFace dataset
print("Loading HuggingFace dataset...")
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

def get_dis(point1, point2):
    dis_x = point1[0] - point2[0]
    dis_y = point1[1] - point2[1]
    dis_z = point1[2] - point2[2]
    return math.sqrt(dis_x * dis_x + dis_y * dis_y + dis_z * dis_z)

def process_proteins_and_create_graphs(df_subset, thresholds=12):
    """Process proteins and create DGL graphs, saving them as dictionaries"""
    import dgl
    import torch
    
    pdb_graphs = {}
    unseen_proteins = set()

    print(f"Processing {len(df_subset)} proteins...")

    for index, row in tqdm(df_subset.iterrows()):
        uni_id, struct_entry = (
            row["protein_id"],
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
            
            # Create DGL graph
            points = valid_coords
            u_list = []
            v_list = []
            dis_list = []
            
            for uid, amino_1 in enumerate(points):
                for vid, amino_2 in enumerate(points):
                    if uid == vid:
                        continue
                    dist = get_dis(amino_1, amino_2)
                    if dist <= thresholds:
                        u_list.append(uid)
                        v_list.append(vid)
                        dis_list.append(dist)
            
            u_list, v_list = torch.tensor(u_list), torch.tensor(v_list)
            dis_list = torch.tensor(dis_list)

            graph = dgl.graph((u_list, v_list), num_nodes=len(points))
            graph.edata["dis"] = dis_list

            esm_features = read_pkl(
                "./processed_file/esm_emds/esm_all.pkl"
            )
            node_features = esm_features[uni_id]
            assert node_features.shape[0] == graph.num_nodes()
            graph.ndata["x"] = torch.from_numpy(node_features)
            pdb_graphs[uni_id] = graph

    print(f"Created {len(pdb_graphs)} graphs")
    print(f"Unseen proteins: {len(unseen_proteins)}")
    
    return pdb_graphs, unseen_proteins

# Process a subset of proteins for testing (you can change this to process all)
print("Processing proteins and creating graphs...")
df_subset = df.head(100)  # Process first 100 proteins for testing
pdb_graphs, unseen_proteins = process_proteins_and_create_graphs(df_subset)

# # Save graphs as dictionary
print("Saving graphs as dictionary...")
os.makedirs('./processed_file/graph_features', exist_ok=True)
save_pkl('./processed_file/graph_features/protein_graphs_dict.pkl', pdb_graphs)

# Save other data
save_pkl('./processed_file/unseen_proteins.pkl', unseen_proteins)

print("Data processing complete!")
print(f"Files saved:")
print(f"  - Protein graphs: ./processed_file/graph_features/protein_graphs_dict.pkl")
print(f"  - Unseen proteins: ./processed_file/unseen_proteins.pkl")

# # Create interpro features
# print("Creating interpro features...")
# interpro_list = read_pkl("interpro_list.pkl")
# inter_idx = {}
# for idx, ipr in enumerate(interpro_list):
#     inter_idx[ipr] = idx

# interpro_dict = {}
# for index, row in df_subset.iterrows():
#     pid = row['protein_id']
#     inters = row['interpro_ids']
#     if inters is not None:
#         inter_matrix = np.zeros(len(interpro_list))
#         for it in inters:
#             if it in inter_idx:
#                 inter_matrix[inter_idx[it]] += 1
#         interpro_dict[pid] = csr_matrix(inter_matrix)

# Save interpro features as dictionary
# save_pkl('./processed_file/protein_interpro_dict.pkl', interpro_dict)
# print(f"Interpro features saved: ./processed_file/protein_interpro_dict.pkl")
# print(f"Created interpro features for {len(interpro_dict)} proteins")
