import json
import sys
import os
import random
import numpy as np
import torch
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hgb.model.HCAN import HCAN
from hgb.utils.utils import LinkPredicator
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score
import scipy.sparse as sp

def run(args):
    def seed_everything(seed=1):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(args.seed)
    if args.dataset == 'lastfm':
        with open('../tmp/HGB/lastfm/info.dat', 'r') as f:
            # read json
            raw_json = f.read()
            info = json.loads(raw_json)
        print(info)
        key2etype = {(int(v['start']), int(v['end'])): int(k) for k, v in info['link.dat'].items()}
        etype2key = {v: k for k, v in key2etype.items()}
        edge_index = [[], []]
        edge_type = []
        with open('../tmp/HGB/lastfm/link.dat', 'r') as f:
            for line in f:
                [u, v, etype, weight] = line.strip().split()
                if u == v:
                    continue
                edge_index[0].append(int(u))
                edge_index[1].append(int(v))
                edge_type.append(int(etype))
                src_type = int(info['link.dat'][etype]['start'])
                tgt_type = int(info['link.dat'][etype]['end'])
                if src_type != tgt_type:
                    if (tgt_type, src_type) not in key2etype.keys():
                        key2etype[(tgt_type, src_type)] = max(key2etype.values()) + 1
                    edge_index[0].append(int(v))
                    edge_index[1].append(int(u))
                    edge_type.append(key2etype[(tgt_type, src_type)])
        edge_index = torch.tensor(edge_index)
        edge_type = torch.tensor(edge_type)
        node_type = []
        with open('../tmp/HGB/lastfm/node.dat', 'r') as f:
            for line in f:
                [node_id, ntype] = line.strip().split()
                node_type.append(int(ntype))
        node_type = torch.tensor(node_type)
        test_edge_index = [[], []]
        test_edge_type = []
        with open('../tmp/HGB/lastfm/link.dat.test', 'r') as f:
            for line in f:
                [u, v, etype, weight] = line.strip().split()
                test_edge_index[0].append(int(u))
                test_edge_index[1].append(int(v))
                test_edge_type.append(int(etype))
        train_edge_index = [[], []]
        valid_edge_index = [[], []]
        train_ratio = 1 - args.valid_ratio

        for etype in edge_type.unique():
            r_mask = (edge_type == etype).nonzero().squeeze()
            r_links = edge_index[:, r_mask]
            r_links = r_links[:, r_links[0].argsort()]
            for src in set(r_links[0].tolist()):
                idx = (r_links[0] == src).nonzero().squeeze()
                if len(idx.shape) == 0:
                    continue
                r_links[1][idx] = r_links[1][idx][r_links[1][idx].argsort()]
            last_hid = -1
            for i in range(0, r_links.size(1)):
                h_id, t_id = r_links[0][i], r_links[1][i]
                if h_id != last_hid:
                    last_hid = h_id
                    train_edge_index[0].append(h_id)
                    train_edge_index[1].append(t_id)
                else:
                    if random.random() < train_ratio:
                        train_edge_index[0].append(h_id)
                        train_edge_index[1].append(t_id)
                    else:
                        valid_edge_index[0].append(h_id)
                        valid_edge_index[1].append(t_id)
        train_edge_index = torch.tensor(train_edge_index)
        valid_edge_index = torch.tensor(valid_edge_index)
        np.random.shuffle(train_edge_index)
        test_edge_index = torch.tensor(test_edge_index)
        test_edge_type = torch.tensor(test_edge_type)
        print(test_edge_type.unique())
        num_nodes_dict = {int(i): (node_type == int(i)).sum() for i in info['node.dat']}
        x_dict = {}
        input_dims_dict = None
        lp_class = 0
        etypes = edge_type.unique().tolist()
        local_node_idx = torch.LongTensor(node_type.size(0))
        local_node_idx[node_type == 0] = torch.arange((node_type == 0).sum())
        local_node_idx[node_type == 1] = torch.arange((node_type == 1).sum())
        local_node_idx[node_type == 2] = torch.arange((node_type == 2).sum())
        edge_feat = torch.ones(edge_index.size(1), dtype=torch.float32)
        edge_attr = torch.stack([edge_type, edge_feat], dim=1)
        homo_data = Data(edge_index=edge_index, edge_attr=edge_attr,
                         node_type=node_type, num_nodes=node_type.size(0), local_node_idx=local_node_idx)

        homo_data = homo_data.to(args.device)
    elif args.dataset == 'pubmed':
        if not os.path.exists('../tmp/HGB/pubmed/pubmed.pkl'):
            node_type = []
            x_dict = {}
            with open('../tmp/HGB/pubmed/node.dat', 'r') as f:
                for line in f:
                    [node_id, node_name, ntype, edmbedding] = line.strip().split()
                    ntype = int(ntype)
                    node_type.append(ntype)
                    embedding = [float(i) for i in edmbedding.split(',')]
                    if ntype not in x_dict.keys():
                        x_dict[ntype] = []
                    x_dict[ntype].append(embedding)
            node_type = torch.tensor(node_type)
            x_dict = {k: torch.tensor(v) for k, v in x_dict.items()}
            input_dims_dict = {}
            for key, x in x_dict.items():
                input_dims_dict[key] = x.size(1)
            key2etype = {}
            edge_index = [[], []]
            edge_weight = []
            edge_type = []
            with open('../tmp/HGB/pubmed/link.dat', 'r') as f:
                for line in f:
                    [u, v, etype, weight] = line.strip().split()
                    if u == v:
                        continue
                    edge_index[0].append(int(u))
                    edge_index[1].append(int(v))
                    edge_type.append(int(etype))
                    edge_weight.append(float(weight))
                    src_type = node_type[int(u)].item()
                    tgt_type = node_type[int(v)].item()
                    if (src_type, tgt_type) not in key2etype.keys():
                        key2etype[(src_type, tgt_type)] = int(etype)
                    if (tgt_type, src_type) not in key2etype.keys():
                        key2etype[(tgt_type, src_type)] = -int(etype)
                    edge_index[0].append(int(v))
                    edge_index[1].append(int(u))
                    edge_type.append(key2etype[(tgt_type, src_type)])
                    edge_weight.append(float(weight))
            edge_index = torch.tensor(edge_index)
            edge_type = torch.tensor(edge_type)
            edge_weight = torch.tensor(edge_weight)
            for k, v in key2etype.items():
                if v < 0:
                    key2etype[k] = max(key2etype.values()) + 1
                    edge_type[edge_type == v] = key2etype[k]
            etype2key = {v: k for k, v in key2etype.items()}
            test_edge_index = [[], []]
            test_edge_type = []
            test_edge_weight = []
            with open('../tmp/HGB/pubmed/link.dat.test', 'r') as f:
                for line in f:
                    [u, v, etype, weight] = line.strip().split()
                    test_edge_index[0].append(int(u))
                    test_edge_index[1].append(int(v))
                    test_edge_type.append(int(etype))
                    test_edge_weight.append(float(weight))
            train_edge_index = [[], []]
            valid_edge_index = [[], []]
            train_ratio = 1 - args.valid_ratio
            for etype in edge_type.unique():
                r_mask = (edge_type == etype).nonzero().squeeze()
                r_links = edge_index[:, r_mask]
                r_links = r_links[:, r_links[0].argsort()]
                for src in set(r_links[0].tolist()):
                    idx = (r_links[0] == src).nonzero().squeeze()
                    if len(idx.shape) == 0:
                        continue
                    r_links[1][idx] = r_links[1][idx][r_links[1][idx].argsort()]
                last_hid = -1
                for i in range(0, r_links.size(1)):
                    h_id, t_id = r_links[0][i], r_links[1][i]
                    if h_id != last_hid:
                        last_hid = h_id
                        train_edge_index[0].append(h_id)
                        train_edge_index[1].append(t_id)
                    else:
                        if random.random() < train_ratio:
                            train_edge_index[0].append(h_id)
                            train_edge_index[1].append(t_id)
                        else:
                            valid_edge_index[0].append(h_id)
                            valid_edge_index[1].append(t_id)
            train_edge_index = torch.tensor(train_edge_index)
            np.random.shuffle(train_edge_index)
            valid_edge_index = torch.tensor(valid_edge_index)
            test_edge_index = torch.tensor(test_edge_index)
            test_edge_type = torch.tensor(test_edge_type)
            test_edge_weight = torch.tensor(test_edge_weight)
            torch.save((node_type, x_dict, input_dims_dict, edge_index, edge_type, edge_weight, etype2key,
                        train_edge_index, valid_edge_index, test_edge_index, test_edge_type, test_edge_weight),
                       '../tmp/HGB/pubmed/pubmed.pkl')
        else:
            node_type, x_dict, input_dims_dict, edge_index, edge_type, edge_weight, etype2key, train_edge_index, valid_edge_index, test_edge_index, test_edge_type, test_edge_weight = torch.load(
                '../tmp/HGB/pubmed/pubmed.pkl')
        print(test_edge_type.unique())
        num_nodes_dict = {int(i): (node_type == int(i)).sum() for i in node_type.unique().tolist()}
        etypes = edge_type.unique().tolist()
        lp_class = 2
        local_node_idx = torch.LongTensor(node_type.size(0))
        for ntype in num_nodes_dict.keys():
            mask = (node_type == ntype).nonzero().squeeze()
            local_node_idx[mask] = torch.arange(mask.size(0))
        edge_attr = torch.stack([edge_type, edge_weight], dim=1)
        homo_data = Data(edge_index=edge_index, edge_attr=edge_attr,
                         node_type=node_type, num_nodes=node_type.size(0), local_node_idx=local_node_idx)
        homo_data = homo_data.to(args.device)

    model = HCAN(num_nodes_dict, input_dim=args.input_dim, hid_dim=args.hid_dim,
                     output_dim=args.hid_dim, etypes=etypes, num_layers=args.num_layers, L=args.L,
                     num_heads=args.num_heads, input_dropout=args.input_dropout, dropout=args.dropout,
                     att_dropout=args.att_dropout,
                     x_ntypes=list(x_dict.keys()), input_dims=input_dims_dict,
                     use_norm=args.use_norm, device=args.device, decoder=args.decoder, c_heads=args.c_heads).to(args.device)
    predicator = LinkPredicator(args.hid_dim, mode=args.decoder).to(args.device)
    if args.decoder == "distmult":
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': predicator.parameters()}], lr=args.lr,
                                     weight_decay=args.decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    def sample_neg(pos):
        s_type, t_type = etype2key[lp_class]
        t_range = (node_type == int(t_type)).nonzero().squeeze()
        t_range = (t_range.min(), t_range.max())
        neg = [[], []]
        for i in range(pos.shape[1]):
            s, t = pos[0][i], pos[1][i]
            if node_type[s] != s_type or node_type[t] != t_type:
                continue
            t = random.randrange(t_range[0], t_range[1])
            neg[0].append(s)
            neg[1].append(t)
        return torch.tensor(neg)

    def sample_two_hop_neg_edges(test_edge_index, all_edge_index, num_nodes):
        test_src_type, test_tgt_type = etype2key[lp_class]

        adj_matrix = sp.coo_matrix(
            (np.ones(all_edge_index.shape[1]), (all_edge_index[0].numpy(), all_edge_index[1].numpy())),
            shape=(num_nodes, num_nodes))
        adj_matrix = adj_matrix.tocsr()

        two_hop_adj = (adj_matrix @ adj_matrix).tolil()

        two_hop_adj[adj_matrix.nonzero()] = 0
        two_hop_adj.setdiag(0)

        test_neigh = [[], []]
        test_labels = []

        head_nodes = test_edge_index[0].unique()
        for head in head_nodes:
            if node_type[head] != test_src_type:
                continue

            pos_list = [[], []]
            pos_list[0] = [head.item()] * (test_edge_index[0] == head).sum().item()
            pos_list[1] = test_edge_index[1][test_edge_index[0] == head].tolist()
            test_neigh[0].extend(pos_list[0])
            test_neigh[1].extend(pos_list[1])
            test_labels.extend([1] * len(pos_list[0]))

            num_pos_edges = len(pos_list[0])

            two_hop_neighbors = np.array(two_hop_adj.rows[head.item()])
            test_mask = (node_type[two_hop_neighbors] == test_tgt_type).numpy()
            test_targets = two_hop_neighbors[test_mask]
            if len(test_targets) > 0:
                sampled_negatives = random.choices(test_targets, k=num_pos_edges)
            else:
                sampled_negatives = []

            test_neigh[0].extend([head.item()] * len(sampled_negatives))
            test_neigh[1].extend(sampled_negatives)
            test_labels.extend([0] * len(sampled_negatives))

        return torch.tensor(test_neigh), torch.tensor(test_labels)

    def train():
        model.train()
        if args.decoder == "distmult":
            predicator.train()
        optimizer.zero_grad()
        out = model(x_dict, homo_data.edge_index, homo_data.edge_attr, homo_data.node_type,
                    homo_data.local_node_idx, args.device)
        train_pos = train_edge_index
        train_neg = sample_neg(train_edge_index)
        pos_labels = torch.tensor([1.0] * train_pos.shape[1], device='cpu')
        neg_labels = torch.tensor([0.0] * train_neg.shape[1], device='cpu')
        train_edge = torch.cat([train_pos, train_neg], dim=1)
        train_labels = torch.cat([pos_labels, neg_labels], dim=0)
        train_scores = torch.sigmoid(predicator.cal_score(out, train_edge))
        loss = predicator.get_loss(train_scores, train_labels)
        if loss.isnan():
            exit()
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def val():
        model.eval()
        predicator.eval()
        out = model(x_dict, homo_data.edge_index, homo_data.edge_attr, homo_data.node_type,
                    homo_data.local_node_idx, args.device)
        val_pos = valid_edge_index
        val_neg = sample_neg(valid_edge_index)
        val_index = torch.cat([val_pos, val_neg], dim=1)
        val_pos_labels = torch.tensor([1.0] * val_pos.shape[1], device='cpu')
        val_neg_labels = torch.tensor([0.0] * val_neg.shape[1], device='cpu')
        val_labels = torch.cat([val_pos_labels, val_neg_labels], dim=0)
        val_scores = torch.sigmoid(predicator.cal_score(out, val_index))
        val_loss = predicator.get_loss(val_scores, val_labels)
        val_roc_auc = roc_auc_score(val_labels, val_scores.cpu())
        val_mrr = predicator.cal_mrr(val_scores, val_index, val_labels)
        return val_roc_auc, val_mrr, val_loss

    test_neigh, test_label = sample_two_hop_neg_edges(test_edge_index,
                                                      torch.cat([train_edge_index, valid_edge_index, test_edge_index],
                                                                dim=1), node_type.size(0))

    @torch.no_grad()
    def test():
        model.eval()
        predicator.eval()
        out = model(x_dict, homo_data.edge_index, homo_data.edge_attr, homo_data.node_type,
                    homo_data.local_node_idx, args.device)
        test_scores = torch.sigmoid(predicator.cal_score(out, test_neigh))
        test_roc_auc = roc_auc_score(test_label, test_scores.cpu())
        test_mrr = predicator.cal_mrr(test_scores, test_neigh, test_label)
        return test_roc_auc, test_mrr

    best_loss = -torch.inf
    best_test_auc, best_test_mrr = 0, 0
    hold = 0
    for epoch in range(args.epochs):
        loss = train()
        val_auc, val_mrr, val_loss = val()
        test_auc, test_mrr = test()
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val_loss: {val_loss:.4f}, "
              f"Val: ROC-AUC={val_auc:.4f}, MRR={val_mrr:.4f}, Test: ROC-AUC={test_auc:.4f}, MRR={test_mrr:.4f}")

        hold += 1
        if -val_loss > best_loss:
            best_loss = -val_loss
            best_test_auc, best_test_mrr = test_auc, test_mrr
            hold = 0
        if hold >= args.patience:
            break
    print("Best Test:{},{}".format(best_test_auc, best_test_mrr))
    with open(f'./records/best_{args.dataset}.txt', 'a') as f:
        f.write(f'Seed:{args.seed}, ROC-AUC:{best_test_auc}, MRR:{best_test_mrr}, '
                f'input_dim:{args.input_dim}, c_heads:{args.c_heads}, hid_dim:{args.hid_dim}, head:{args.num_heads}, '
                f'num_layers:{args.num_layers}, L:{args.L}, lr:{args.lr}, decay:{args.decay}, use_norm:{args.use_norm}, '
                f'input_dropout: {args.input_dropout}, dropout:{args.dropout}, att_dropout:{args.att_dropout}\n')
