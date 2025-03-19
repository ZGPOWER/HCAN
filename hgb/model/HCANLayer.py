from hgb.model.Conv import *
import torch.nn.functional as F


class HCANLayer(nn.Module):
    def __init__(self, in_feat: int, hid_dim: int, ntypes: int, num_rels: int, num_heads: int, nlayers=None,
                 dropout: float = 0.5, att_dropout=0.0,
                 negative_slope: float = 0.1,
                 use_norm=False, device='cpu', c_heads=1):
        super(HCANLayer, self).__init__()
        self.num_heads = num_heads
        self.ntypes=ntypes
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
                SGCConv(in_channels=hid_dim, out_channels=hid_dim, num_node_types=ntypes, num_edge_types=num_rels,
                         use_norm=use_norm, c_heads=c_heads)
            )
            for _ in range(1, nlayers - 1):
                self.layers.append(
                    SGCConv(in_channels=hid_dim, out_channels=hid_dim, num_node_types=ntypes, num_edge_types=num_rels,
                             use_norm=use_norm, c_heads=c_heads)
                )
            self.layers.append(MixConv(in_channels=hid_dim*self.nlayers, out_channels=hid_dim, num_node_types=ntypes,
                                       num_edge_types=num_rels, heads=num_heads, use_norm=use_norm, att_dropout=att_dropout, c_heads=c_heads))
        else:
            self.layers.append(MixConv(in_channels=hid_dim, out_channels=hid_dim, num_node_types=ntypes,
                                       num_edge_types=num_rels, heads=num_heads, use_norm=use_norm, att_dropout=att_dropout, c_heads=c_heads))
        self.feat_lins = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(ntypes)])
        self.root_lins = nn.ModuleList([nn.Linear(in_feat, hid_dim) for _ in range(ntypes)])
        if self.use_norm:
            self.norm_1 = nn.LayerNorm(hid_dim*self.nlayers)
            self.norm_2 = nn.LayerNorm(hid_dim)
        self.act = nn.LeakyReLU()
        self.dp = nn.Dropout(p=self.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        for layer in self.layers:
            layer.reset_parameters()
        for lin in self.feat_lins:
            nn.init.xavier_normal_(lin.weight,gain=gain)
            nn.init.zeros_(lin.bias)
        for lin in self.root_lins:
            nn.init.xavier_normal_(lin.weight,gain=gain)
            nn.init.zeros_(lin.bias)

    def proj_x(self, x, node_type, root=False):
        outs = []
        for i in range(self.ntypes):
            if root:
                outs.append(self.root_lins[i](x))
            else:
                outs.append(self.feat_lins[i](x))
        outs = torch.stack(outs, dim=1)
        node_type_expanded = node_type.unsqueeze(1).unsqueeze(2).expand(-1, 1, outs.size(2))
        proj = outs.gather(dim=1, index=node_type_expanded).squeeze(1)
        return proj

    def forward(self, x, edge_index, edge_attr, node_type):
        x = self.proj_x(x, node_type, True)
        tmp = [x]
        for i in range(self.nlayers-1):
            x = self.layers[i](x, edge_index, edge_attr, node_type)
            tmp.append(self.proj_x(x, node_type))
        x = torch.concat(tmp, dim=1)
        if self.use_norm:
            x = self.norm_1(x)
        x = self.dp(x)
        x = self.act(x)
        x = self.layers[-1](x, edge_index, edge_attr, node_type)
        if self.use_norm:
            x = self.norm_2(x)
        return x
