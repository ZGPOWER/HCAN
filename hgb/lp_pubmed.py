import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from run_lp import run

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--device', type=str, default='cuda:0')
ap.add_argument('--num_layers', type=int, default=4)
ap.add_argument('--L', type=int, default=1)
ap.add_argument('--c_heads', type=int, default=1)
ap.add_argument('--input_dim', type=int, default=256)
ap.add_argument('--hid_dim', type=int, default=128)
ap.add_argument('--num_heads', type=int, default=2)
ap.add_argument('--lr', type=float, default=0.0005)
ap.add_argument('--input_dropout', type=float, default=0.0)
ap.add_argument('--dropout', type=float, default=0.6)
ap.add_argument('--att_dropout', type=float, default=0.0)
ap.add_argument('--decay', type=float, default=0.000)
ap.add_argument('--use_norm', type=bool, default=True)
ap.add_argument('--epochs', type=int, default=8000)
ap.add_argument('--patience', type=int, default=200)
ap.add_argument('--valid_ratio', type=int, default=0.1)
ap.add_argument('--decoder', type=str, default='distmult')
ap.add_argument('--dataset', type=str, default='pubmed')
args = ap.parse_args()

run(args)
