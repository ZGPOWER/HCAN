import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import LGConv
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from torch_sparse import SparseTensor

class MultiChannelProj(nn.Module):
    def __init__(self, channels, in_dim, out_dim, bias=True):
        super(MultiChannelProj, self).__init__()
        self.channels = channels
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.Tensor(self.channels, self.in_dim, self.out_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.channels, self.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)
        gain = nn.init.calculate_gain('relu')
        xavier_uniform_(self.weight, gain=gain)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return torch.einsum('bcn,cnm->bcm', x, self.weight) + self.bias


class HeteroSelfAttention(nn.Module):
    def __init__(self, hid_dim, num_heads):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        assert self.hid_dim % (4*self.num_heads) == 0
        self.K = nn.Linear(self.hid_dim, self.hid_dim//8)
        self.Q = nn.Linear(self.hid_dim, self.hid_dim//8)
        self.V = nn.Linear(self.hid_dim, self.hid_dim)
        self.act = nn.LeakyReLU(0.2)
        self.att_drop = nn.Dropout(0.0)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.reset_parameters()

    def reset_parameters(self):
        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.K.weight, gain=gain)
        xavier_uniform_(self.Q.weight, gain=gain)
        xavier_uniform_(self.V.weight, gain=gain)
        nn.init.zeros_(self.K.bias)
        nn.init.zeros_(self.Q.bias)
        nn.init.zeros_(self.V.bias)

    def forward(self, x):
        B, M, C = x.size()
        H = self.num_heads
        q = self.Q(x).view(B, M, H, -1).permute(0,2,1,3)
        k = self.K(x).view(B, M, H, -1).permute(0,2,3,1)
        v = self.V(x).view(B, M, H, -1).permute(0,2,1,3)
        beta = F.softmax(self.act(q @ k / math.sqrt(q.size(-1))), dim=-1)
        beta = self.att_drop(beta)
        output = self.gamma*(beta@v)
        return output.transpose(1, 2).reshape(B,M,C) + x


class DecoupledHCOAT(nn.Module):
    def __init__(self, num_feats, in_dim, hid_dim, num_classes, num_edge_types, num_heads, input_dims, args, out_dim=None, output=True):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_edge_types = num_edge_types
        self.num_paths = num_feats
        self.num_classes = num_classes
        if any([v != in_dim for k, v in input_dims.items()]):
            self.embedings = nn.ParameterDict({})
            for k, v in input_dims.items():
                if v != in_dim:
                    self.embedings[k] = nn.Parameter(
                        torch.Tensor(v, in_dim).uniform_(-0.5, 0.5))
        else:
            self.embedings = None

        self.input_drop = nn.Dropout(args.input_dropout)
        self.feat_layers = nn.Sequential(
            MultiChannelProj(self.num_paths, in_dim, hid_dim),
            nn.LayerNorm([self.num_paths, hid_dim]),
            nn.PReLU(),
            nn.Dropout(args.dropout),
            MultiChannelProj(self.num_paths, hid_dim, hid_dim),
            nn.LayerNorm([self.num_paths, hid_dim]),
            nn.PReLU(),
            nn.Dropout(args.dropout),
        )
        self.att_layers = HeteroSelfAttention(hid_dim, num_heads)
        self.merge_channels = nn.Linear(self.num_paths*self.hid_dim, self.hid_dim)
        self.residual = nn.Linear(in_dim, hid_dim, bias=False)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(args.dropout)
        if out_dim is None:
            self.out_dim = num_classes
        else:
            self.out_dim = out_dim
        self.output = output
        if self.output:
            self.output_layers = nn.Sequential(
                nn.Linear(hid_dim, hid_dim, bias=False),
                nn.BatchNorm1d(hid_dim),
                nn.PReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(hid_dim, self.out_dim, bias=False),
                nn.BatchNorm1d(self.out_dim)
            )
            self.label_residual = nn.Sequential(
                nn.Linear(num_classes, hid_dim, bias=False),
                nn.BatchNorm1d(hid_dim),
                nn.PReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(hid_dim, hid_dim, bias=False),
                nn.BatchNorm1d(hid_dim),
                nn.PReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(hid_dim, hid_dim, bias=False),
                nn.BatchNorm1d(hid_dim),
                nn.PReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(hid_dim, self.out_dim, bias=True)
            )

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for layer in self.feat_layers:
            if isinstance(layer, MultiChannelProj):
                layer.reset_parameters()

        nn.init.xavier_uniform_(self.merge_channels.weight, gain=gain)
        nn.init.zeros_(self.merge_channels.bias)
        nn.init.xavier_uniform_(self.residual.weight, gain=gain)
        if self.output:
            for layer in self.output_layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            for layer in self.label_residual:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, batch_feats, tgt_type, label_emb):
        if self.embedings is not None:
            for k, v in batch_feats.items():
                if k in self.embedings:
                    batch_feats[k] = v @ self.embedings[k]
        tgt_feat = self.input_drop(batch_feats[tgt_type])
        x = self.input_drop(torch.stack(list(batch_feats.values()), dim=1))
        x = self.feat_layers(x)
        x = self.att_layers(x)
        x = self.merge_channels(x.view(x.size(0),-1))
        x = x + self.residual(tgt_feat)
        if self.output:
            x = self.dropout(self.prelu(x))
            x = self.output_layers(x)
            x = x + self.label_residual(label_emb)
        return x

class MyGAT(nn.Module):
    def __init__(self, num_feats, num_label_feats, in_dim, hid_dim, num_classes, num_edge_types, num_heads, input_dims, args):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.feat_layer = DecoupledHCOAT(num_feats, in_dim, hid_dim, num_classes, num_edge_types, num_heads, input_dims, args, out_dim=hid_dim, output=False)
        assert self.hid_dim % (4 * self.num_heads) == 0
        self.K = nn.Linear(self.hid_dim, self.hid_dim // 4)
        self.Q = nn.Linear(self.hid_dim, self.hid_dim // 4)
        self.V = nn.Linear(self.hid_dim, self.hid_dim)
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.beta = nn.Parameter(torch.tensor([1.]))
        self.activation = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(args.dropout),
        )
        self.output_layers = nn.Sequential(
            nn.Linear(hid_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hid_dim, num_classes, bias=False),
            nn.BatchNorm1d(num_classes)
        )
        self.label_residual = nn.Sequential(
            nn.Linear(num_classes, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hid_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hid_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(hid_dim, num_classes, bias=True)
        )
        self.label_weights = nn.Parameter(torch.tensor([1/num_label_feats]*num_label_feats))
        self.reset_parameters()

    def reset_parameters(self):
        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)
        self.feat_layer.reset_parameters()
        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.K.weight, gain=gain)
        xavier_uniform_(self.Q.weight, gain=gain)
        xavier_uniform_(self.V.weight, gain=gain)
        nn.init.zeros_(self.K.bias)
        nn.init.zeros_(self.Q.bias)
        nn.init.zeros_(self.V.bias)
        for layer in self.output_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        for layer in self.label_residual:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self,batch_feats, tgt_type, batch_label_feats, edge_index, batch_label_embeds):
        feat = self.feat_layer(batch_feats, tgt_type, batch_label_embeds)
        x = self.activation(feat)
        k = self.K(x[edge_index[0]])
        q = self.Q(x[edge_index[1]])
        v = self.V(x)
        k = k.view(k.size(0), self.num_heads, -1)
        q = q.view(q.size(0), self.num_heads, -1)
        scores = F.leaky_relu((k*q).sum(dim=-1) / math.sqrt(k.shape[-1]))
        scores = scores.sum(dim=1)
        weights = softmax(scores, edge_index[1])
        (row,col) = edge_index
        adj_t = SparseTensor(row=col,col=row, value=weights.squeeze())
        output = adj_t.matmul(v, reduce='mean')
        x = self.gamma * output + x[:output.shape[0]]
        x = self.output_layers(x)
        label_embed =  torch.sum(torch.stack([self.label_weights[i]*v for i, v in enumerate(list(batch_label_feats.values()))]), dim=0)
        x = self.beta * x + self.label_residual(label_embed[:output.shape[0]])
        return x