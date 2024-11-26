import sys
import os
import random
import time
import numpy as np
import torch.cuda.amp as amp
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F
import torch
import torch.nn as nn
from hgb.model.HCAN import HCAN
from hgb.utils.utils import FocalLoss
from torch_geometric.data import Data
from torch_geometric.datasets.hgb_dataset import HGBDataset
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.utils import remove_self_loops
from sklearn.metrics import f1_score
import torch_geometric.transforms as T


def run(args):
    def seed_everything(seed=1):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

    seed_everything(args.seed)

    if args.to_undirected:
        transform = T.Compose([T.ToUndirected(merge=True)])
        dataset = HGBDataset(root='../../tmp/HGB', name=args.dataset, transform=transform)
    else:
        dataset = HGBDataset(root='../../tmp/HGB', name=args.dataset)
    data = dataset[0]
    print(data)
    nc_target_type = args.nc_target_type
    num_classes = args.num_classes

    edge_index_dict = data.edge_index_dict
    num_nodes_dict = data.num_nodes_dict
    if args.dataset == 'Freebase':
        x_dict = {}
    else:
        x_dict = data.x_dict
    if args.dataset == 'ACM':
        for k, v in edge_index_dict.items():
            if k[0] == k[-1]:
                edge_index_dict[k] = remove_self_loops(v)[0]

    out = group_hetero_graph(edge_index_dict, num_nodes_dict)
    edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out
    edge_feat = torch.ones(edge_index.size(1), dtype=torch.float32)
    edge_attr = torch.stack([edge_type, edge_feat], dim=1)
    homo_data = Data(edge_index=edge_index, edge_attr=edge_attr,
                     node_type=node_type, local_node_idx=local_node_idx,
                     num_nodes=node_type.size(0))
    if args.dataset == 'IMDB':
        homo_data.y = node_type.new_full((node_type.size(0), 5), -1).squeeze().float()
    else:
        homo_data.y = node_type.new_full((node_type.size(0), 1), -1).squeeze().long()
    homo_data.y[local2global[nc_target_type]] = data.y_dict[nc_target_type]
    homo_data_mask = {}
    train_valid_indices = np.where(data[nc_target_type]['train_mask'] == True)[0]
    np.random.shuffle(train_valid_indices)
    split_index = int((1.0 - args.valid_ratio) * len(train_valid_indices))
    train_indices = np.sort(train_valid_indices[:split_index])
    valid_indices = np.sort(train_valid_indices[split_index:])
    homo_data_mask['train_mask'] = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data_mask['train_mask'][local2global[nc_target_type][train_indices]] = True
    homo_data_mask['valid_mask'] = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data_mask['valid_mask'][local2global[nc_target_type][valid_indices]] = True
    homo_data_mask['test_mask'] = torch.zeros((node_type.size(0)), dtype=torch.bool)
    homo_data_mask['test_mask'][local2global[nc_target_type][data[nc_target_type]['test_mask']]] = True
    homo_data = homo_data.to(args.device)

    if args.dataset == 'Freebase':
        x_dict = {}
        num_nodes_dict = {key2int[k]: v for k, v in num_nodes_dict.items()}
        input_dims_dict = None
    else:
        x_dict = {key2int[k]: x_dict[k] for k, v in num_nodes_dict.items() if k in x_dict}
        num_nodes_dict = {key2int[k]: v for k, v in num_nodes_dict.items()}
        input_dims_dict = {k: v.size(1) for k, v in x_dict.items()}
    x_dict = {k: v.to(args.device) for k, v in x_dict.items()}
    etypes = list(edge_index_dict.keys())
    model = HCAN(num_nodes_dict, input_dim=args.input_dim, hid_dim=args.hid_dim,
                     output_dim=num_classes, etypes=etypes, num_layers=args.num_layers, L=args.L,
                     num_heads=args.num_heads, dropout=args.dropout, att_dropout=args.att_dropout,
                     x_ntypes=list(x_dict.keys()), input_dims=input_dims_dict, use_norm=args.use_norm,
                     device=args.device,
                     input_dropout=args.input_dropout, c_heads=args.c_heads).to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    model_parameters = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                         'weight_decay': args.decay},
                        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                         'weight_decay': 0.0}]
    optimizer = torch.optim.Adam(model_parameters, lr=args.lr, weight_decay=args.decay)
    scaler = amp.GradScaler()
    if args.dataset in ['DBLP']:
        loss_func = FocalLoss(gamma=1)
    else:
        loss_func = nn.BCEWithLogitsLoss()
    def train():
        model.train()
        optimizer.zero_grad()
        train_mask = homo_data_mask['train_mask']
        out = model(x_dict, homo_data.edge_index, homo_data.edge_attr, homo_data.node_type,
                    homo_data.local_node_idx, args.device)
        out = out[train_mask]
        y = homo_data.y[train_mask]
        if args.dataset != "IMDB":
            labels = F.one_hot(y, num_classes).float()
        else:
            labels = y
        loss = loss_func(out.float(), labels.float())
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def test():
        model.eval()
        out = model(x_dict, homo_data.edge_index, homo_data.edge_attr, homo_data.node_type,
                    homo_data.local_node_idx, args.device)
        macros = []
        micros = []
        for split in ['train_mask', 'valid_mask', 'test_mask']:
            homo_mask = homo_data_mask[split]
            if args.dataset!="IMDB":
                pred = out[homo_mask].argmax(dim=-1, keepdim=True).cpu()
            else:
                pred = (torch.sigmoid(out[homo_mask]) > 0.5).float().cpu()
            labels = homo_data.y[homo_mask].cpu()
            micros.append(f1_score(labels, pred, average='micro'))
            macros.append(f1_score(labels, pred, average='macro'))
            if split == 'valid_mask':
                if args.dataset != "IMDB":
                    val_labels = F.one_hot(labels, num_classes).float()
                else:
                    val_labels = labels
                valid_loss = loss_func(out[homo_mask].cpu(), val_labels)
        return macros, micros, valid_loss

    best_loss = -torch.inf
    best_val = 0
    best_test_macro, best_test_micro = 0, 0
    hold = 0
    for epoch in range(args.epochs):
        loss, t, embeds = train()
        [train_macro, valid_macro, test_macro], [train_micro, valid_micro, test_micro], valid_loss = test()

        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},'
              f'Train Macro: ({train_macro:.4f}, {train_micro:.4f}), Valid Macro: ({valid_macro:.4f}, {valid_micro:.4f}), '
              f'Test Macro: ({test_macro:.4f}, {test_micro:.4f}), Valid_loss: {valid_loss:.4f}')
        hold += 1
        if args.stopping == 'loss':
            if -valid_loss > best_loss:
                best_loss = -valid_loss
                hold = 0
                best_test_macro = test_macro
                best_test_micro = test_micro
        else:
            if best_val < (valid_micro+valid_macro):
                best_val = valid_micro+valid_macro
                best_loss = -valid_loss
                hold = 0
                best_test_macro = test_macro
                best_test_micro = test_micro
            elif best_val==(valid_micro+valid_macro) and -valid_loss>best_loss:
                best_loss = -valid_loss
                hold=0
                best_test_macro = test_macro
                best_test_micro = test_micro
        if hold >= args.patience:
            break
    print("Best Macro-F1:{} Micro-F1:{}".format(best_test_macro, best_test_micro))
    with open(f'./records/best_{args.dataset.lower()}.txt', 'a') as f:
        f.write(f'Seed:{args.seed}, Macro-F1:{best_test_macro}, Micro-F1:{best_test_micro}, '
                f'input_dim:{args.input_dim}, hid_dim:{args.hid_dim}, head:{args.num_heads}, num_layers:{args.num_layers}, '
                f'L:{args.L}, lr:{args.lr}, c_heads:{args.c_heads}, input_dropout:{args.input_dropout}, '
                f'dropout: {args.dropout}, att_dropout:{args.att_dropout}, decay:{args.decay}, pat:{args.patience}, '
                f'use_norm:{args.use_norm}\n')
