import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import degree, to_networkx
from collections import defaultdict
import random
import numpy as np
import torch_geometric.transforms as T
from tqdm import tqdm
import math


class LinkPredicator(nn.Module):
    def __init__(self, hid_dim, mode='dot'):
        super(LinkPredicator, self).__init__()
        self.mode = mode
        if mode == 'distmult':
            self.w_relation = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        self.mode = mode

    def reset_parameters(self):
        if self.mode == 'distmult':
            nn.init.xavier_normal_(self.w_relation, gain=1.414)

    def cal_score(self, embeddings, edge_index):
        s = embeddings[edge_index[0]]
        o = embeddings[edge_index[1]]
        if self.mode == 'dot':
            return (s * o).sum(dim=1)
        elif self.mode == 'distmult':
            r = self.w_relation
            r = r.to(embeddings.device)
            s = torch.einsum('ij,jk->ik', s, r)
            return torch.einsum('ik,ik->i', s, o)
        else:
            raise NotImplementedError

    def get_loss(self, scores, labels):
        return F.binary_cross_entropy(scores.cpu(), labels.cpu())

    def cal_mrr(self, scores, edge_index, labels):
        scores = scores.cpu()
        edge_index = edge_index.cpu()
        labels = labels.cpu()
        mrr_list, cur_mrr = [], 0
        t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(list), defaultdict(list)
        for i, h_id in enumerate(edge_index[0]):
            h_id = h_id.item()
            t_dict[h_id].append(edge_index[1][i].item())
            labels_dict[h_id].append(labels[i].item())
            conf_dict[h_id].append(scores[i].item())
        for h_id in t_dict.keys():
            conf_array = np.array(conf_dict[h_id])
            rank = np.argsort(-conf_array)
            sorted_label_array = np.array(labels_dict[h_id])[rank]

            pos_index = np.where(sorted_label_array == 1)[0]
            if len(pos_index) == 0:
                continue
            pos_min_rank = np.min(pos_index)
            cur_mrr = 1 / (1 + pos_min_rank)
            mrr_list.append(cur_mrr)
        if len(mrr_list) == 0:
            return 0
        mrr = np.mean(mrr_list)
        return mrr


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss