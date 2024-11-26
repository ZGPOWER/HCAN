import gc
from typing import Optional
import torch
from sphinx.registry import merge_source_suffix
from torch import nn
from torch_geometric.nn.inits import constant
from torch_sparse import SparseTensor
from torch_scatter import scatter_add,scatter_mean
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.utils import softmax, get_laplacian, add_self_loops
from torch_geometric.nn.conv import RGCNConv
import torch.nn.functional as F
import math


class RelConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types,
                 num_edge_types, use_norm=False, c_heads=1):
        super(RelConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.use_norm = use_norm
        self.c_heads = c_heads

        self.weight = nn.ParameterList([torch.Tensor(c_heads, out_channels, in_channels) for _ in range(self.num_edge_types)])

        self.root_lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for weight in self.weight:
            for c in range(self.c_heads):
                nn.init.xavier_uniform_(weight[c], gain=gain)
        for lin in self.root_lins:
            nn.init.xavier_normal_(lin.weight, gain=gain)

    def forward(self, x, edge_index, edge_attr, node_type):
        edge_type, edge_weight = edge_attr[:, 0], edge_attr[:, 1]
        out = x.new_zeros(x.size(0), self.out_channels)
        for i in range(self.num_edge_types):
            mask = edge_type == i
            edge_index_masked = edge_index[:, mask]
            edge_weight_masked = edge_weight[mask]
            out.add_(self.propagate(edge_index_masked, x=x, edge_weight=edge_weight_masked, edge_type=i))
        for i in range(self.num_node_types):
            mask = node_type == i
            out[mask] += self.root_lins[i](x[mask])
        return out

    def message(self, x_j, edge_weight, edge_type):
        outputs = torch.bmm(x_j.unsqueeze(0).expand(self.c_heads,-1,-1),self.weight[edge_type].permute(0,2,1))
        return edge_weight.view(-1,1) * outputs.sum(dim=0)


class CoA(MessagePassing):
    def __init__(self, in_channels, out_channels, num_node_types, num_edge_types, use_norm=True, negative_slope=0.2,
                 heads=8, att_dropout=0.0, c_heads=1):
        super(CoA, self).__init__(node_dim=0, aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.use_norm = use_norm
        self.neg_slope = negative_slope
        self.n_heads = heads
        self.norms = nn.ModuleList()
        self.att_softmax = None
        self.att_dropout = att_dropout
        self.c_heads = c_heads

        att_dim = out_channels
        self.W_k = nn.ModuleList([nn.Linear(in_channels, att_dim, bias=False) for _ in range(num_node_types)])
        self.W_q = nn.ModuleList([nn.Linear(in_channels, att_dim, bias=False) for _ in range(num_node_types)])
        self.d_k = att_dim//heads
        self.root_lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=True)
            for _ in range(num_node_types)
        ])
        self.W_v = nn.ParameterList([torch.Tensor(self.c_heads, out_channels//2, in_channels) for _ in range(self.num_edge_types)])
        self.att_fc = nn.ParameterList([torch.Tensor(self.c_heads, 1, 2*self.d_k) for _ in range(self.num_edge_types)])
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for lin in self.root_lins:
            nn.init.xavier_normal_(lin.weight, gain=gain)
        for lin in self.W_k:
            nn.init.xavier_normal_(lin.weight, gain=gain)
        for lin in self.W_q:
            nn.init.xavier_normal_(lin.weight, gain=gain)
        for weight in self.W_v:
            for c in range(self.c_heads):
                nn.init.xavier_normal_(weight[c], gain=gain)
        for weight in self.att_fc:
            for c in range(self.c_heads):
                nn.init.xavier_normal_(weight[c], gain=gain)

    def forward(self, x, edge_index, edge_attr, node_type):
        assert x.shape[0] == node_type.shape[0]
        edge_type, edge_weight = edge_attr[:, 0], edge_attr[:, 1]
        src_x = torch.zeros(x.size(0), self.W_k[0].weight.size(0), device=x.device)
        dst_x = torch.zeros(x.size(0), self.W_q[0].weight.size(0), device=x.device)
        for i in range(self.num_node_types):
            mask = node_type == i
            src_x[mask] = self.W_k[i](x[mask]).to(src_x.dtype)
            dst_x[mask] = self.W_q[i](x[mask]).to(src_x.dtype)
        out = x.new_zeros(x.size(0), self.out_channels)
        for i in range(self.num_edge_types):
            mask = edge_type == i
            edge_index_masked = edge_index[:, mask]
            edge_weight_masked = edge_weight[mask]
            out.add_(self.propagate(edge_index_masked, x=x, src_x=src_x, dst_x=dst_x, edge_weight=edge_weight_masked, edge_type=i).flatten(1))
        for i in range(self.num_node_types):
            mask = node_type == i
            out[mask] += self.root_lins[i](x[mask])
        return out

    def message(self, edge_index_i, x_j, src_x_j, dst_x_i, edge_weight, edge_type):
        B,D=src_x_j.size()
        feat_src = src_x_j.view(-1, self.n_heads, self.d_k).unsqueeze(0).expand(self.c_heads,-1,-1,-1)
        feat_dst = dst_x_i.view(-1, self.n_heads, self.d_k).unsqueeze(0).expand(self.c_heads,-1,-1,-1)
        att = (torch.cat([feat_dst, feat_src], dim=-1)*self.att_fc[edge_type].unsqueeze(1)).sum(dim=3)
        att = softmax(att, edge_index_i, dim=1)
        rev_att = 1-att
        att_dropmask = F.dropout(torch.ones_like(att), p=self.att_dropout, training=self.training)
        att = att * att_dropmask
        rev_att = rev_att * att_dropmask
        rel_feat = torch.bmm(x_j.unsqueeze(0).expand(self.c_heads,-1,-1),self.W_v[edge_type].permute(0,2,1)).view(self.c_heads, B, self.n_heads, -1)
        msg = rel_feat * att.view(self.c_heads, -1, self.n_heads, 1)
        rev_msg = rel_feat * rev_att.view(self.c_heads, -1, self.n_heads, 1)
        msg = torch.cat([msg, rev_msg],dim=-1)
        msg = edge_weight.view(-1, 1, 1) * msg.sum(dim=0)
        return msg
