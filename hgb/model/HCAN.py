from copy import copy
from hgb.model.HCANLayer import HCANLayer
import torch
import torch.nn as nn
import torch.nn.functional as F


class HCAN(nn.Module):
    def __init__(self, num_nodes_dict: dict, input_dim, hid_dim: int, output_dim: int, etypes: list,
                 num_layers: int = 2, L: int = 1, num_heads: int = 8, dropout: float = 0.5, att_dropout=0.5,
                 x_ntypes=None, input_dims=None, use_norm=False, device='cpu',
                 negative_slope: float = 0.2, input_dropout: float = 0.0, decoder='proj', c_heads=1):
        super(HCAN, self).__init__()
        self.num_nodes_dict = num_nodes_dict
        self.num_rels = len(etypes)
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.L = L
        self.num_heads = num_heads
        self.c_heads = c_heads
        self.input_dims = input_dims
        self.device = device
        self.dropout = dropout
        self.use_norm = use_norm
        self.negative_slope = negative_slope
        self.input_dropout = input_dropout
        self.decoder = decoder
        emb_keys = set(num_nodes_dict.keys()).difference(set(x_ntypes)) if x_ntypes else set(num_nodes_dict.keys())
        self.emb_dict = nn.ParameterDict({
            f'{key}': nn.Parameter(torch.Tensor(num_nodes_dict[key], input_dim))
            for key in emb_keys
        })
        self.layers = nn.ModuleList()
        self.layers.append(HCANLayer(input_dim, hid_dim, ntypes=len(list(num_nodes_dict.keys())),
                                           num_rels=self.num_rels,
                                           num_heads=num_heads, dropout=dropout, att_dropout=att_dropout,
                                           negative_slope=negative_slope,
                                           nlayers=num_layers, use_norm=use_norm, device=device,
                                           c_heads=c_heads))
        for _ in range(1, L):
            self.layers.append(HCANLayer(hid_dim, hid_dim, ntypes=len(list(num_nodes_dict.keys())),
                                               num_rels=self.num_rels, num_heads=num_heads, dropout=dropout,
                                               att_dropout=att_dropout,
                                               negative_slope=negative_slope, nlayers=num_layers,
                                               use_norm=use_norm, device=device,
                                               c_heads=c_heads))

        self.input_linears = nn.ModuleDict()
        if input_dims is not None:
            for key in x_ntypes:
                self.input_linears[str(key)] = nn.Linear(input_dims[key], input_dim, bias=True)
        if self.decoder == 'proj':
            self.proj = nn.Linear(hid_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(input_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for key in self.emb_dict.keys():
            nn.init.xavier_normal_(self.emb_dict[key], gain=gain)
        for key in self.input_linears.keys():
            nn.init.xavier_normal_(self.input_linears[key].weight, gain=gain)
        for layer in self.layers:
            layer.reset_parameters()
        if self.decoder == 'proj':
            nn.init.xavier_normal_(self.proj.weight, gain=gain)

    def group_input(self, x_dict, node_type, local_node_idx, device):
        h = torch.zeros((node_type.size(0), self.input_dim), device=device)

        for key, x in x_dict.items():
            if x.device != device:
                x = x.to(device)
            mask = node_type == key
            if x.size(1) != self.input_dim:
                x = self.input_linears[str(key)](x)
            if self.use_norm:
                x = self.norm(x)
            h[mask] = x[local_node_idx[mask].to(device)]

        for key, emb in self.emb_dict.items():
            if emb.device != device:
                emb = emb.to(device)
            mask = node_type == int(key)
            if self.use_norm:
                emb = self.norm(emb)
            h[mask] = emb[local_node_idx[mask].to(device)]

        return h

    def forward(self, x_dict, edge_index, edge_attr, node_type, local_node_idx, device):
        x = self.group_input(x_dict, node_type, local_node_idx, device)
        x = F.dropout(x, p=self.input_dropout, training=self.training)
        if edge_index.device != device:
            edge_index = edge_index.to(device)
            node_type = node_type.to(device)
            edge_attr = edge_attr.to(device)

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr, node_type)
            if i != len(self.layers) - 1:
                x = F.leaky_relu(x, negative_slope=self.negative_slope)
        if self.decoder == 'proj':
            x = self.proj(x)
        return x