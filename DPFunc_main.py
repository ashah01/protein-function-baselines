from ruamel.yaml import YAML
from logzero import logger
from pathlib import Path
import warnings

import torch
import numpy as np
from dgl.dataloading import GraphDataLoader

from DPFunc.data_utils import get_pdb_data, get_mlb, get_inter_whole_data
from DPFunc.models import combine_inter_model
from DPFunc.objective import AverageMeter
from DPFunc.model_utils import test_performance_gnn_inter, merge_result, FocalLoss
from DPFunc.evaluation import new_compute_performance_deepgoplus

import os
import pickle as pkl
import click
from tqdm.auto import tqdm
from datasets import load_dataset

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

    train_data = load_dataset("wanglab/cafa5", "cafa5_reasoning", split="train").to_pandas()
    valid_data = load_dataset("wanglab/cafa5", "cafa5_reasoning", split="test").to_pandas()

    key = "go_ids"
    if data_cnf['name'] == 'bp':
        key = "go_bp"
    elif data_cnf['name'] == 'cc':
        key = "go_cc"
    elif data_cnf['name'] == 'mf':
        key = "go_mf"
    
    train_data = train_data.dropna(subset=[key, "interpro_ids", "structure_path"])
    valid_data = valid_data.dropna(subset=[key, "interpro_ids", "structure_path"])

    train_go = [x.tolist() for x in train_data[key]]
    valid_go = [x.tolist() for x in valid_data[key]]

    train_pid_list = train_data["protein_id"].tolist()
    valid_pid_list = valid_data["protein_id"].tolist()

    with open("interpro_list.pkl", "rb") as f:
        interpro_dict = pkl.load(f)

    train_interpro_list = [[interpro_dict[ip] for ip in x.tolist()] for x in train_data["interpro_ids"]]
    valid_interpro_list = [[interpro_dict[ip] for ip in x.tolist()] for x in valid_data["interpro_ids"]]
    train_interpro = get_inter_whole_data(train_interpro_list, "./processed_file/interpro_matrix_train.pkl")
    valid_interpro = get_inter_whole_data(valid_interpro_list, "./processed_file/interpro_matrix_valid.pkl")

    train_graphs = get_pdb_data(pdb_graph_file = "./processed_file/graph_features/protein_graphs_dict_train.pkl", pid_list = train_pid_list)
    logger.info('train data done')
    valid_graphs = get_pdb_data(pdb_graph_file = "./processed_file/graph_features/protein_graphs_dict_valid.pkl", pid_list = valid_pid_list)
    logger.info('valid data done')

    mlb = get_mlb(train_go)
    labels_num = len(mlb.classes_)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        train_y = mlb.transform(train_go).astype(np.float32)
        valid_y = mlb.transform(valid_go).astype(np.float32)

    idx_goid = {}
    goid_idx = {}
    for idx, goid in enumerate(mlb.classes_):
        idx_goid[idx] = goid
        goid_idx[goid] = idx

    train_data = []
    for i in range(len(train_y)):
        if train_graphs[i] is not None:
            train_data.append((train_graphs[i], i, train_y[i]))

    train_dataloader = GraphDataLoader(
        train_data,
        batch_size=64,
        drop_last=False,
        shuffle=True)

    valid_data = []
    for i in range(len(valid_y)):
        if valid_graphs[i] is not None:
            valid_data.append((valid_graphs[i], i, valid_y[i]))

    valid_dataloader = GraphDataLoader(
        valid_data,
        batch_size=64,
        drop_last=False,
        shuffle=False)

    del train_graphs
    del valid_graphs
    
    logger.info('Loading Data & Model')
    
    model = combine_inter_model(inter_size=36062, 
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
        for batched_graph, sample_idx, labels in tqdm(train_dataloader, leave=False):
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            # * Since we've dropped the nan interpros, the only constraint on rows is the validity of the contact maps
            # * so it's actually okay if i just use protein_id indexed interpro_ids
            inter_features = (torch.from_numpy(train_interpro[sample_idx].indices).to(device).long(), 
                            torch.from_numpy(train_interpro[sample_idx].indptr).to(device).long(), 
                            torch.from_numpy(train_interpro[sample_idx].data).to(device).float())
            feats = batched_graph.ndata['x']

            logits = model(inter_features, batched_graph, feats)

            loss = loss_fn(logits, labels)
            train_loss_vals.update(loss.item(), len(labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        plus_fmax, plus_aupr, plus_t, df, valid_loss_avg = test_performance_gnn_inter(model, valid_dataloader, valid_pid_list, valid_interpro, valid_y, idx_goid, goid_idx, ont, device)
        logger.info('Epoch: {}, Train Loss: {:.6f}\tValid Loss: {:.6f}, plus_Fmax on valid: {:.4f}, AUPR on valid: {:.4f}, cut-off: {:.2f}, df_shape: {}'.format(e, 
                                                                                                                                df.shape))

        if e > min(used_model_performance):
                replace_ind = np.where(used_model_performance==min(used_model_performance))[0][0]
                used_model_performance[replace_ind] = e
                torch.save({'epoch': e,'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}, 
                        './save_models/{0}_{1}_{2}of{3}model.pt'.format(pre_name, ont, replace_ind, 3))
                logger.info("\t\t\t\t\tSave")

if __name__ == '__main__':
    main()