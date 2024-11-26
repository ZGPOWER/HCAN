import time
import math
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.utils import to_undirected, k_hop_subgraph, degree
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.data import GraphSAINTRandomWalkSampler, Data
import torch_geometric.transforms as T
import torch.nn.functional as F
from tqdm import tqdm
import random
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn.conv import LGConv
from torch_sparse import SparseTensor
from model import DecoupledHCOAT, MyGAT
import sparse_tools
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--input_dim', type=int, default=256)
ap.add_argument('--hid_dim', type=int, default=512)
ap.add_argument('--num_heads', type=int, default=1)
ap.add_argument('--lr', type=float, default=0.001)
ap.add_argument('--input_dropout', type=float, default=0.1)
ap.add_argument('--dropout', type=float, default=0.5)
ap.add_argument('--att_dropout', type=float, default=0.)
ap.add_argument('--decay', type=float, default=0.000)
ap.add_argument('--epochs', type=int, default=2000)
ap.add_argument('--batch_size', type=int, default=10000)
ap.add_argument('--num_neighbors', type=int, default=5)
ap.add_argument('--patience', type=int, default=100)
ap.add_argument('--num_classes', type=int, default=349)
args = ap.parse_args()
print(args)

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
set_random_seed(args.seed)
cur_timestamp = time.time()
dataset = OGB_MAG('../tmp/OGB_MAG')
data = dataset[0].to('cpu')
data = T.ToUndirected()(data)
train_nid = torch.nonzero(data['paper'].train_mask).squeeze()
val_nid = torch.nonzero(data['paper'].val_mask).squeeze()
test_nid = torch.nonzero(data['paper'].test_mask).squeeze()
evaluator = Evaluator(name='ogbn-mag')

raw_x_dict = data.x_dict
raw_num_nodes_dict = data.num_nodes_dict
edge_index_dict = data.edge_index_dict.copy()
print(edge_index_dict.keys())
print(data)
p2p_index = edge_index_dict[('paper', 'cites', 'paper')]

out = group_hetero_graph(edge_index_dict, raw_num_nodes_dict)
edge_index, edge_type, node_type, local_node_idx, local2global, key2int = out

edge_attr = torch.stack([edge_type, torch.ones_like(edge_type)], dim=1)

homo_data = Data(edge_index=edge_index, edge_attr=edge_attr,
                 node_type=node_type, local_node_idx=local_node_idx,
                 num_nodes=node_type.size(0))

homo_data.y = torch.zeros(node_type.size(0), args.num_classes).squeeze()
homo_data.y[local2global['paper']] = F.one_hot(data.y_dict['paper']).float()

homo_data.train_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
homo_data.train_mask[local2global['paper'][train_nid]] = True
homo_data.valid_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
homo_data.valid_mask[local2global['paper'][val_nid]] = True
homo_data.test_mask = torch.zeros((node_type.size(0)), dtype=torch.bool)
homo_data.test_mask[local2global['paper'][test_nid]] = True
homo_data = homo_data.to('cpu')

x_dict = {}
for key, x in raw_x_dict.items():
    x_dict[key[0].upper()] = x
num_nodes_dict = {}
label_dict = {}
for key, N in raw_num_nodes_dict.items():
    num_nodes_dict[key[0].upper()] = N
    if key[0].upper() not in x_dict:
        x_dict[key[0].upper()] = torch.Tensor(N, args.input_dim).uniform_(-0.5, 0.5)
    label_dict[key[0].upper()] = torch.zeros(N, args.num_classes)
label_dict['P'][train_nid] = homo_data.y[homo_data.train_mask]
etypes = list(edge_index_dict.keys())
tgt_type='P'


def graph_conv(x_dict, hops):
    x_data = {}
    for key, x in x_dict.items():
        x_data[key] = {key: x}
    for hop in range(hops):
        for etype, adj in edge_index_dict.items():
            stype, dtype = etype[0][0].upper(), etype[-1][0].upper()
            new_x = []
            for k, v in x_data[stype].items():
                if len(k) != hop + 1:
                    continue
                (row, col) = adj
                adj_t = SparseTensor(row=col, col=row)
                tmp = adj_t.matmul(v, reduce='mean')
                new_x.append((k + dtype, tmp))
            for k, v in new_x:
                x_data[dtype][k] = v
    return x_data
x_data = graph_conv(x_dict, args.num_layers)
label_data = graph_conv(label_dict, args.num_layers)
for k,v in label_data.items():
    removes = []
    for k2,v2 in v.items():
        if len(k2)==1 or k2[0]!=tgt_type:
            removes.append(k2)
    for k2 in removes:
        label_data[k].pop(k2)
print(x_data[tgt_type].keys(), label_data[tgt_type].keys())
for k,v in label_data[tgt_type].items():
    if len(k)==3 and k[0]==k[2]:
        for etype in edge_index_dict:
            if etype[0][0].upper()==tgt_type and etype[-1][0].upper()==k[1]:
                row,col = edge_index_dict[etype]
                adj=SparseTensor(row=row,col=col)
                print(k)
                if k[0]==k[1]:
                    adj = adj.to_symmetric()
                    assert torch.all(adj.get_diag()==0)
                    diag = sparse_tools.spspmm_diag_sym_AAA(adj)
                else:
                    diag = sparse_tools.spspmm_diag_sym_ABA(adj)
                label_data[tgt_type][k] -= diag.unsqueeze(-1)*label_dict[tgt_type]
                break
feats = {k: v for k,v in x_data[tgt_type].items()}
label_feats = {k: v for k,v in label_data[tgt_type].items()}
label_emb = torch.zeros(feats['P'].size(0), args.num_classes)
for _,v in label_feats.items():
    label_emb += v
label_emb /= len(label_feats)
init2sort = torch.cat([train_nid, val_nid, test_nid])
newcode = torch.zeros_like(init2sort)
for i, j in enumerate(init2sort):
    newcode[j] = i
p2p_index = newcode[p2p_index]
p2pdata = Data(edge_index=p2p_index, num_nodes=p2p_index.max().item()+1)
feats = {k: v[init2sort] for k,v in feats.items()}
label_feats = {k: v[init2sort] for k,v in label_feats.items()}
label_emb = label_emb[init2sort]
train_nodes_num = train_nid.size(0)
val_nodes_num = val_nid.size(0)
test_nodes_num = test_nid.size(0)
total_nodes_num = train_nodes_num+val_nodes_num+test_nodes_num
train_loader = NeighborLoader(p2pdata, batch_size=args.batch_size, input_nodes=torch.arange(train_nodes_num), shuffle=True, num_neighbors=[args.num_neighbors])
eval_loader = NeighborLoader(p2pdata, batch_size=args.batch_size, input_nodes=torch.arange(train_nodes_num, total_nodes_num), shuffle=False, num_neighbors=[args.num_neighbors])
labels = data.y_dict['paper'][init2sort].to(args.device)
num_path = len(list(feats.keys()))
num_label_path = len(list(label_feats.keys()))
input_dims = {k:v.size(-1) for k,v in feats.items()}
model = MyGAT(num_path, num_label_path, args.input_dim, args.hid_dim, args.num_classes, len(etypes), args.num_heads, input_dims, args).to(args.device)
print(model)
def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
print("# Params:", get_n_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
loss_func = nn.CrossEntropyLoss()
def train():
    model.train()
    total_loss = 0
    total_examples = 0
    for batch in train_loader:
        batch_feats = {k: v[batch.n_id].to(args.device) for k, v in feats.items()}
        batch_label_feats = {k: v[batch.n_id].to(args.device) for k, v in label_feats.items()}
        batch_label_embeds = label_emb[batch.n_id].to(args.device)
        batch_edge_index = batch.edge_index.to(args.device)
        optimizer.zero_grad()
        out = model(batch_feats, tgt_type, batch_label_feats, batch_edge_index, batch_label_embeds)
        pred = out
        y = labels[batch.n_id][:pred.shape[0]]
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_examples += 1
    return total_loss / total_examples


def eval():
    with torch.no_grad():
        model.eval()
        outputs = []
        for batch in eval_loader:
            batch_feats = {k: v[batch.n_id].to(args.device) for k, v in feats.items()}
            batch_label_feats = {k: v[batch.n_id].to(args.device) for k, v in label_feats.items()}
            batch_label_embeds = label_emb[batch.n_id].to(args.device)
            batch_edge_index = batch.edge_index.to(args.device)
            optimizer.zero_grad()
            out = model(batch_feats, tgt_type, batch_label_feats, batch_edge_index, batch_label_embeds)
            pred = out
            outputs.append(pred)
        output = torch.concat(outputs, dim=0)
        y_pred = output.argmax(dim=-1)
        assert y_pred.size(0) == val_nodes_num + test_nodes_num
        val_pred = y_pred[:val_nodes_num]
        test_pred = y_pred[val_nodes_num:total_nodes_num]
        val_label = labels[train_nodes_num:train_nodes_num + val_nodes_num]
        test_label = labels[train_nodes_num + val_nodes_num:total_nodes_num]
        val_acc = evaluator.eval({
            "y_true": val_label.view(-1, 1),
            "y_pred": val_pred.view(-1, 1),
        })["acc"]
        test_acc = evaluator.eval({
            "y_true": test_label.view(-1, 1),
            "y_pred": test_pred.view(-1, 1),
        })["acc"]
        return (val_acc, test_acc)


best_loss = float('inf')
best_acc = 0
best_test_acc = 0
patience = 0
for epoch in tqdm(range(1, args.epochs + 1)):
    loss = train()
    print("Loss: ", loss)
    acc = eval()
    print(f"Acc: Valid {acc[0]}, Test {acc[1]}")
    if acc[0] > best_acc:
        best_acc = acc[0]

        best_test_acc = acc[1]
        torch.save(model, f'checkpoints/mag_{cur_timestamp}.pt')
        patience = 0
    patience += 1
    if patience >= args.patience:
        break
model = torch.load(f'checkpoints/mag_{cur_timestamp}.pt')
acc = eval()
print(f"Best epoch: Valid acc {acc[0]}, Test acc {acc[1]}")
