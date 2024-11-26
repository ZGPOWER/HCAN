from hgb.model.Conv import *
import torch.nn.functional as F


class HCANLayer(nn.Module):
    def __init__(self, in_feat: int, hid_dim: int, ntypes: int, num_rels: int, num_heads: int, nlayers=None,
                 dropout: float = 0.5, att_dropout=0.0,
                 negative_slope: float = 0.1,
                 use_norm=False, device='cpu', c_heads=1):
        super(HCANLayer, self).__init__()
        self.num_heads = num_heads
        self.nlayers = nlayers
        self.layers = nn.ModuleList()
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.device = device
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.in_feat = in_feat
        self.hid_dim = hid_dim
        self.use_norm = use_norm

        if nlayers > 1:
            self.layers.append(
                RelConv(in_channels=in_feat, out_channels=hid_dim, num_node_types=ntypes, num_edge_types=num_rels,
                         use_norm=use_norm, c_heads=c_heads))
            for _ in range(1, nlayers - 1):
                self.layers.append(
                    RelConv(in_channels=hid_dim, out_channels=hid_dim, num_node_types=ntypes, num_edge_types=num_rels,
                             use_norm=use_norm, c_heads=c_heads))
            self.layers.append(CoA(in_channels=hid_dim, out_channels=hid_dim, num_node_types=ntypes,
                                       num_edge_types=num_rels, heads=num_heads, use_norm=use_norm, att_dropout=att_dropout, c_heads=c_heads))
        else:
            self.layers.append(CoA(in_channels=in_feat, out_channels=hid_dim, num_node_types=ntypes,
                                       num_edge_types=num_rels, heads=num_heads, use_norm=use_norm, att_dropout=att_dropout, c_heads=c_heads))
        if self.use_norm:
            self.norm = nn.LayerNorm(hid_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr, node_type):
        for i, layer in enumerate(self.layers):
            if x.device != self.device:
                x = x.to(self.device)
            out = layer(x, edge_index, edge_attr, node_type)
            x = out
            if self.use_norm:
                x = self.norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if i != len(self.layers) - 1:
                x = F.leaky_relu(x, negative_slope=self.negative_slope)
        return x

