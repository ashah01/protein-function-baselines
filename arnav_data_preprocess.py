import torch
import dgl
import os
from tqdm.auto import tqdm
from Bio.PDB.MMCIFParser import FastMMCIFParser
import math
import gzip
import pickle as pkl
from load_cafa5 import load_cafa5_dataset

# Load HuggingFace dataset
print("Loading HuggingFace dataset...")
train_dataset, val_dataset, test_dataset = load_cafa5_dataset(
    dataset="wanglab/cafa5",
    dataset_name="cafa5_reasoning",
    dataset_subset=None,
    max_length=2048,
    val_split_ratio=0.1,
    seed=23,
    return_as_chat_template=False,
    cache_dir='/Users/arnavshah/Code/DPFunc/cafa5',
    structure_dir='/Users/arnavshah/Code/DPFunc/cafa5/extracted' # '/Users/arnavshah/Code/DPFunc/cafa5/extracted'
)

def read_pkl(file_path):
    with open(file_path,'rb') as fr:
        return pkl.load(fr)

def save_pkl(file_path, val):
    fw = open(file_path, 'wb')
    pkl.dump(val, fw)
    fw.close()

def extract_sequence_and_ca_coords(pdb_file, chain_id=None):
    parser = FastMMCIFParser(QUIET=True)
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


def get_dis(point1, point2):
    dis_x = point1[0] - point2[0]
    dis_y = point1[1] - point2[1]
    dis_z = point1[2] - point2[2]
    return math.sqrt(dis_x * dis_x + dis_y * dis_y + dis_z * dis_z)

def process_proteins_and_create_graphs(
    df_subset,
    thresholds: int = 12,
    output_dir: str = "./processed_file/graph_features/train",
):
    """Process proteins and create DGL graphs, saving *each* graph to disk as soon as it is built.

    This avoids holding all graphs in memory at once and greatly reduces peak RAM usage on large
    datasets.

    Returns
    -------
    saved_graph_paths : list[str]
        Files that were written for successfully processed proteins.
    unseen_proteins : set[str]
        Proteins that could not be processed because of missing structure files.
    """

    os.makedirs(output_dir, exist_ok=True)

    unseen_proteins = set()
    pdb_graphs = {}
    print(f"Processing {len(df_subset)} proteins...")

    # Load the ESM embeddings once to avoid re-loading inside the loop.
    protein_map = read_pkl("./large_storage/goodarzilab/ashah/esm_train_protein_map.pkl")

    for row in tqdm(df_subset, desc="Proteins"):
        uni_id, struct_entry = row["protein_id"], row["structure_path"]

        # Skip examples without structure information
        if struct_entry is None:
            unseen_proteins.add(uni_id)
            continue

        pdb_file = struct_entry

        if not os.path.exists(pdb_file):
            print(f"Warning: structure file not found for {uni_id} → {pdb_file}")
            unseen_proteins.add(uni_id)
            continue

        coords_list = extract_sequence_and_ca_coords(pdb_file, "A")

        if not coords_list:
            unseen_proteins.add(uni_id)
            continue

        valid_coords = [coord for coord in coords_list if coord is not None]

        # Build graph
        points = valid_coords
        u_list, v_list, dis_list = [], [], []

        for uid, amino_1 in enumerate(points):
            for vid, amino_2 in enumerate(points):
                if uid == vid:
                    continue
                dist = get_dis(amino_1, amino_2)
                if dist <= thresholds:
                    u_list.append(uid)
                    v_list.append(vid)
                    dis_list.append(dist)

        u_list = torch.tensor(u_list)
        v_list = torch.tensor(v_list)
        dis_list = torch.tensor(dis_list)

        graph = dgl.graph((u_list, v_list), num_nodes=len(points))
        graph.edata["dis"] = dis_list

        esm_file_idx = protein_map[uni_id]
        esm_features = read_pkl(f"/large_storage/goodarzilab/ashah/esm_emds_train/esm_part_{esm_file_idx}.pkl")

        node_features = esm_features[uni_id]

        graph.ndata["x"] = torch.from_numpy(node_features)

        # Save graph to disk immediately
        pdb_graphs[uni_id] = graph

    print(f"Unseen proteins: {len(unseen_proteins)}")

    return pdb_graphs, unseen_proteins

# ============================= RUN PRE-PROCESSING =============================

print("Processing proteins and creating graphs (memory-efficient mode)...")

pdb_graphs, unseen_proteins = process_proteins_and_create_graphs(train_dataset)

# Persist list of graph paths and unseen proteins for downstream loading
os.makedirs("./processed_file/graph_features", exist_ok=True)

save_pkl("./processed_file/graph_features/train_graph_paths.pkl", pdb_graphs)
save_pkl("./processed_file/unseen_proteins.pkl", unseen_proteins)

print("Data processing complete!")
print("Files saved:")
print("  - Per-protein graphs: ./processed_file/graph_features/train/*.bin")
print("  - Graph path index : ./processed_file/graph_features/train_graph_paths.pkl")
print("  - Unseen proteins  : ./processed_file/unseen_proteins.pkl")

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
