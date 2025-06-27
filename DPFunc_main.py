from ruamel.yaml import YAML
from logzero import logger
from pathlib import Path
import warnings

import torch
import numpy as np
from dgl.dataloading import GraphDataLoader

from DPFunc.data_utils import get_mlb
from DPFunc.models import combine_inter_model
from DPFunc.objective import AverageMeter
from DPFunc.model_utils import test_performance_gnn_inter, merge_result, FocalLoss
from DPFunc.evaluation import new_compute_performance_deepgoplus

import os
import pickle as pkl
import click
from tqdm.auto import tqdm
from datasets import load_dataset
from scipy.sparse import csr_matrix

class ProteinDataset(torch.utils.data.Dataset):
    """Custom dataset for protein data with graphs, interpro features, and GO terms"""
    def __init__(self, data, labels):
        self.data = data  # List of (graph, interpro, go_terms) tuples
        self.labels = labels  # Pre-computed labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        graph, interpro, go_terms = self.data[idx]
        label = self.labels[idx]
        return graph, interpro, label

def load_data_dicts():
    """Load all data dictionaries - graphs, ESM embeddings, and interpro features"""
    logger.info("Loading data dictionaries...")
    
    # Load graphs
    graph_file = './processed_file/graph_features/protein_graphs_dict.pkl'
    if os.path.exists(graph_file):
        with open(graph_file, 'rb') as f:
            graph_dict = pkl.load(f)
        logger.info(f"Loaded {len(graph_dict)} graphs")
    else:
        logger.warning(f"Graph file not found: {graph_file}")
        graph_dict = {}
    
    # Load ESM embeddings
    esm_file = './processed_file/esm_emds/esm_all.pkl'
    if os.path.exists(esm_file):
        with open(esm_file, 'rb') as f:
            esm_dict = pkl.load(f)
        logger.info(f"Loaded {len(esm_dict)} ESM embeddings")
    else:
        logger.warning(f"ESM file not found: {esm_file}")
        esm_dict = {}
    
    # Load interpro features
    interpro_file = './processed_file/protein_interpro_dict.pkl'
    if os.path.exists(interpro_file):
        with open(interpro_file, 'rb') as f:
            interpro_dict = pkl.load(f)
        logger.info(f"Loaded {len(interpro_dict)} interpro features")
    else:
        logger.warning(f"Interpro file not found: {interpro_file}")
        interpro_dict = {}
    
    return graph_dict, esm_dict, interpro_dict

def build_dataset_from_df(df, graph_dict, esm_dict, interpro_dict, go_key):
    """Build dataset from HuggingFace DataFrame with key-based lookups"""
    data = []
    go_terms = []
    valid_pids = []
    
    missing_graph = 0
    missing_esm = 0
    missing_interpro = 0
    
    for _, row in df.iterrows():
        pid = row['protein_id']
        
        # Check if all required data is available
        if pid not in graph_dict:
            missing_graph += 1
            continue
        if pid not in esm_dict:
            missing_esm += 1
            continue
        if pid not in interpro_dict:
            missing_interpro += 1
            continue
        
        graph = graph_dict[pid]
        esm_features = esm_dict[pid]
        interpro_features = interpro_dict[pid]
        go = row[go_key]
        
        # Verify ESM features match graph nodes
        if esm_features.shape[0] != graph.num_nodes():
            continue
        
        # Update graph with ESM features
        graph.ndata['x'] = torch.from_numpy(esm_features)
        
        data.append((graph, interpro_features, go))
        go_terms.append(go)
        valid_pids.append(pid)
    
    logger.info(f"Dataset building complete:")
    logger.info(f"  - Missing graphs: {missing_graph}")
    logger.info(f"  - Missing ESM: {missing_esm}")
    logger.info(f"  - Missing interpro: {missing_interpro}")
    logger.info(f"  - Valid samples: {len(data)}")
    
    return data, go_terms, valid_pids

def test_performance_gnn_inter_new(model, dataloader, valid_pids, valid_y, idx_goid, goid_idx, ont, device):
    """Updated validation function for the new data format"""
    model.eval()
    valid_loss_vals = AverageMeter()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batched_graph, batched_interpro, labels in tqdm(dataloader, leave=False):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            
            # Convert interpro features
            inter_indices = []
            inter_indptr = []
            inter_data = []
            
            for interpro in batched_interpro:
                inter_indices.append(torch.from_numpy(interpro.indices).long())
                inter_indptr.append(torch.from_numpy(interpro.indptr).long())
                inter_data.append(torch.from_numpy(interpro.data).float())
            
            inter_indices = torch.cat(inter_indices).to(device)
            inter_indptr = torch.cat(inter_indptr).to(device)
            inter_data = torch.cat(inter_data).to(device)
            
            inter_features = (inter_indices, inter_indptr, inter_data)
            
            feats = batched_graph.ndata['x']
            logits = model(inter_features, batched_graph, feats)
            
            loss = FocalLoss()(logits, labels)
            valid_loss_vals.update(loss.item(), len(labels))
            
            predictions = torch.sigmoid(logits)
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all predictions and labels
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Calculate metrics (you may need to implement this based on your evaluation needs)
    # For now, returning placeholder values
    plus_fmax = 0.0
    plus_aupr = 0.0
    plus_t = 0.0
    df = None
    
    return plus_fmax, plus_aupr, plus_t, df, valid_loss_vals.avg

@click.command()
@click.option('-d', '--data-cnf', type=click.Choice(['bp', 'mf', 'cc']))
@click.option('-n', '--gpu-number', type=click.INT, default=0)
@click.option('-e', '--epoch-number', type=click.INT, default=15)
@click.option('-p', '--pre-name', type=click.STRING, default='temp_model')

def main(data_cnf, gpu_number, epoch_number, pre_name):
    yaml = YAML(typ='safe')
    ont = data_cnf
    data_cnf, model_cnf = yaml.load(Path('./configure/{}.yaml'.format(data_cnf))), yaml.load(Path('./configure/dgg.yaml'))
    device = torch.device('cuda:{}'.format(gpu_number))

    data_name, model_name = data_cnf['name'], model_cnf['name'] 
    run_name = F'{model_name}-{data_name}'
    logger.info('run_name: {}'.format(run_name))

    data_cnf['mlb'] = Path(data_cnf['mlb'])
    data_cnf['results'] = Path(data_cnf['results'])
    logger.info(F'Model: {model_name}, Dataset: {data_name}')

    # Load HuggingFace dataset
    df = load_dataset("wanglab/cafa5", "cafa5_reasoning", split="train").to_pandas()
    key = "go_ids"
    if data_cnf['name'] == 'bp':
        key = "go_bp"
    elif data_cnf['name'] == 'cc':
        key = "go_cc"
    elif data_cnf['name'] == 'mf':
        key = "go_mf"
    
    df = df.dropna(subset=[key])
    
    # Split data into train/valid (you may want to implement proper splitting)
    # For now, using first 80% as train, last 20% as valid
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    valid_df = df.iloc[split_idx:]
    
    logger.info(f'Train samples: {len(train_df)}, Valid samples: {len(valid_df)}')

    # Load all data dictionaries
    graph_dict, esm_dict, interpro_dict = load_data_dicts()
    
    # Check if we have enough data
    if len(graph_dict) == 0:
        logger.error("No graphs available. Please run the data preprocessing scripts first.")
        return
    
    if len(esm_dict) == 0:
        logger.error("No ESM embeddings available. Please run the ESM processing script first.")
        return
    
    if len(interpro_dict) == 0:
        logger.error("No interpro features available. Please run the data preprocessing script first.")
        return
    
    # Build datasets
    logger.info('Building train dataset...')
    train_data, train_go_terms, train_pids = build_dataset_from_df(
        train_df, graph_dict, esm_dict, interpro_dict, key
    )
    
    logger.info('Building valid dataset...')
    valid_data, valid_go_terms, valid_pids = build_dataset_from_df(
        valid_df, graph_dict, esm_dict, interpro_dict, key
    )
    
    # Check if we have enough data
    if len(train_data) == 0:
        logger.error("No training data available. Please check your data files.")
        return
    
    if len(valid_data) == 0:
        logger.error("No validation data available. Please check your data files.")
        return
    
    logger.info(f"Final dataset sizes - Train: {len(train_data)}, Valid: {len(valid_data)}")
    
    # Create MLB and transform labels
    mlb = get_mlb(Path(data_cnf['mlb']), train_go_terms)
    labels_num = len(mlb.classes_)
    
    # Transform labels
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_y = mlb.transform(train_go_terms).astype(np.float32)
        valid_y = mlb.transform(valid_go_terms).astype(np.float32)

    # Create index mappings
    idx_goid = {}
    goid_idx = {}
    for idx, goid in enumerate(mlb.classes_):
        idx_goid[idx] = goid
        goid_idx[goid] = idx
        
    # Create data loaders with custom dataset
    train_dataset = ProteinDataset(train_data, train_y)
    valid_dataset = ProteinDataset(valid_data, valid_y)
    
    train_dataloader = GraphDataLoader(
        train_dataset,
        batch_size=64,
        drop_last=False,
        shuffle=True)

    valid_dataloader = GraphDataLoader(
        valid_dataset,
        batch_size=64,
        drop_last=False,
        shuffle=False)

    logger.info('Loading Data & Model')
    
    # Get interpro feature size from first sample
    interpro_size = train_data[0][1].shape[1] if train_data else 36062
    
    model = combine_inter_model(inter_size=interpro_size, 
                                inter_hid=1280, 
                                graph_size=1280, 
                                graph_hid=1280, 
                                label_num=labels_num, head=4).to(device)
    logger.info(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)
    loss_fn = FocalLoss()

    used_model_performance = np.array([-1.0]*3)

    for e in range(epoch_number):
        model.train()
        train_loss_vals = AverageMeter()
        for batched_graph, batched_interpro, labels in tqdm(train_dataloader, leave=False):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            
            # Convert interpro features to the format expected by the model
            # Create sparse tensor format for interpro features
            # This assumes your model expects (indices, indptr, data) format
            inter_indices = []
            inter_indptr = []
            inter_data = []
            
            for interpro in batched_interpro:
                inter_indices.append(torch.from_numpy(interpro.indices).long())
                inter_indptr.append(torch.from_numpy(interpro.indptr).long())
                inter_data.append(torch.from_numpy(interpro.data).float())
            
            # Concatenate all interpro features
            inter_indices = torch.cat(inter_indices).to(device)
            inter_indptr = torch.cat(inter_indptr).to(device)
            inter_data = torch.cat(inter_data).to(device)
            
            inter_features = (inter_indices, inter_indptr, inter_data)
            
            feats = batched_graph.ndata['x']

            logits = model(inter_features, batched_graph, feats)

            loss = loss_fn(logits, labels)
            train_loss_vals.update(loss.item(), len(labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation
        plus_fmax, plus_aupr, plus_t, df, valid_loss_avg = test_performance_gnn_inter_new(
            model, valid_dataloader, valid_pids, valid_y, idx_goid, goid_idx, ont, device
        )
        logger.info('Epoch: {}, Train Loss: {:.6f}\tValid Loss: {:.6f}, plus_Fmax on valid: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}'.format(
            e, train_loss_vals.avg, valid_loss_avg, plus_fmax, plus_aupr, plus_t
        ))

        if e > min(used_model_performance):
                replace_ind = np.where(used_model_performance==min(used_model_performance))[0][0]
                used_model_performance[replace_ind] = e
                torch.save({'epoch': e,'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                        './save_models/{0}_{1}_{2}of{3}model.pt'.format(pre_name, ont, replace_ind, 3))
                logger.info("\t\t\t\t\tSave")

if __name__ == '__main__':
    main()
