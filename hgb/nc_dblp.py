import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from train import run

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--num_layers', type=int, default=2)
ap.add_argument('--L', type=int, default=2)
ap.add_argument('--c_heads', type=int, default=1)
ap.add_argument('--input_dim', type=int, default=512)
ap.add_argument('--hid_dim', type=int, default=512)
ap.add_argument('--num_heads', type=int, default=4)
ap.add_argument('--lr', type=float, default=0.0007)
ap.add_argument('--input_dropout', type=float, default=0.)
ap.add_argument('--dropout', type=float, default=0.7)
ap.add_argument('--att_dropout', type=float, default=0.)
ap.add_argument('--decay', type=float, default=0.000)
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--use_norm', type=bool, default=True)
ap.add_argument('--patience', type=int, default=20)
ap.add_argument('--stopping', type=str, default='f1')
ap.add_argument('--valid_ratio', type=int, default=0.2)
ap.add_argument('--dataset', type=str, default='DBLP')
ap.add_argument('--nc_target_type', type=str, default='author')
ap.add_argument('--num_classes', type=int, default=4)
ap.add_argument('--to_undirected', type=bool, default=False)
args = ap.parse_args()

run(args)
