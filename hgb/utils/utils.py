import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from collections import defaultdict
import numpy as np

class EarlyStopping:
    def __init__(self,patience,path:list):
        self.hold=0
        self.patience=patience
        self.stop=False
        self.best_score=None
        self.path=path
    def __call__(self, score, model:list):
        if self.best_score is None:
            self.best_score=score
            for i,m in enumerate(model):
                torch.save(m.state_dict(),self.path[i])
        elif score>self.best_score:
            self.best_score=score
            self.hold=0
            for i,m in enumerate(model):
                torch.save(m.state_dict(),self.path[i])
        else:
            self.hold+=1
            if self.hold>=self.patience:
                self.stop=True


class LinkPredicator(nn.Module):
    def __init__(self, hid_dim, mode='dot'):
        super(LinkPredicator, self).__init__()
        self.mode = mode
        if mode == 'distmult':
            self.w_relation = nn.Parameter(torch.Tensor(hid_dim, hid_dim))

    def reset_parameters(self):
        if self.mode == 'distmult':
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(self.w_relation, gain=gain)

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


class Classifier(nn.Module):
    def __init__(self, hid_dim, num_classes):
        super(Classifier, self).__init__()

        self.projection_layer = nn.Linear(hid_dim, num_classes)

    def forward(self, node_representations):
        """
        :param node_representations:
        :return:
        """
        class_scores = self.projection_layer(node_representations)
        return F.log_softmax(class_scores, dim=-1)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss


def find_edge_type(data: Data, src_node, tgt_node, key2int, node_types, edge_types):
    int2key_n = {v: k for k, v in key2int.items() if k in node_types}
    int2key_e = {v: k for k, v in key2int.items() if k in edge_types}
    src_type = int2key_n[data.node_type[src_node].item()]
    tgt_type = int2key_n[data.node_type[tgt_node].item()]
    for key in edge_types:
        if src_type == key[0] and tgt_type == key[-1]:
            return key2int[key]
    new_type = (src_type, 'to', tgt_type)
    cur_max = max(int2key_e.keys())
    key2int[new_type] = cur_max + 1
    return cur_max + 1


def homo2hetero(num_nodes_dict, edge_index_dict):
    hetero_data = HeteroData()
    for key, x in num_nodes_dict.items():
        hetero_data[key].x = torch.randn(x, 128)
    for key, edge_index in edge_index_dict.items():
        hetero_data[key].edge_index = edge_index
    return hetero_data
