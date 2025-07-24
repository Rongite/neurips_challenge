import torch
from pathlib import Path
import argparse
from functools import partial

# --- Setup paths and import from full_pipeline ---
from full_pipeline import (
    ListDataset, GRIT_DIR, ROOT, DATA_ROOT, GRAPH_DIR, RESULTS_DIR, TARGETS, NULL_FOR_SUB,
    build_graph_cache, consolidate, subset_graphs, weighted_mae, ensure_poly_loader
)
import sys
sys.path.append(str(GRIT_DIR))
from torch_geometric.graphgym.config import cfg as gym_cfg, set_cfg, load_cfg, dump_cfg
from yacs.config import CfgNode
from grit.transform.posenc_stats import compute_posenc_stats
from grit.transform.transforms import pre_transform_in_memory

# --- Main Debug Logic ---
import yaml

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True, help='Path to base YAML config')
args = ap.parse_args()

with open(ROOT / args.cfg) as f:
    base_cfg = yaml.safe_load(f)

t_graphs_list = torch.load(GRAPH_DIR / 'test_graphs.pt', weights_only=False)
t_graphs = ListDataset(t_graphs_list)

pe_enabled_list = []
if base_cfg.get('posenc_RRWP', {}).get('enable', False):
    pe_enabled_list.append('RRWP')

if pe_enabled_list:
    print("--- Before pre_transform_in_memory ---")
    for i in range(len(t_graphs)):
        print(f'Graph {i}: {t_graphs.get(i)}')

    temp_cfg = CfgNode()
    set_cfg(temp_cfg)
    temp_cfg.set_new_allowed(True)
    temp_cfg.merge_from_file(ROOT / args.cfg)
    temp_cfg.dataset.node_encoder_num_types = 119 # From previous logs
    temp_cfg.dataset.edge_encoder_num_types = 4   # From previous logs
    is_undirected = all(d.is_undirected() for d in t_graphs_list[:10])
    pre_transform_in_memory(t_graphs, partial(compute_posenc_stats, pe_types=pe_enabled_list,
                                    is_undirected=is_undirected, cfg=temp_cfg), show_progress=True)

    print("\n--- After pre_transform_in_memory ---")
    for i in range(len(t_graphs)):
        print(f'Graph {i}: {t_graphs.get(i)}')
        if not hasattr(t_graphs.get(i), 'x') or t_graphs.get(i).x is None:
            print(f'Graph {i} has lost its x attribute.')
