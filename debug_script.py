import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / 'data'
SUPP_DIR = DATA_ROOT / 'train_supplement'
GRAPH_DIR = SUPP_DIR / 'graphs'

t_graphs_list = torch.load(GRAPH_DIR / 'test_graphs.pt', weights_only=False)

for i, g in enumerate(t_graphs_list):
    print(f'Graph {i}: {g}')
    if not hasattr(g, 'x') or g.x is None:
        print(f'Graph {i} has no x attribute or x is None.')
