#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze the saved PyG graphs (data/train_graphs.pt):
- Compute overall stats: average/min/max nodes & edges
- Optionally print top-5 largest/smallest molecules by node count
"""

import sys
from pathlib import Path
import torch
from collections import Counter

# --------------------------------------
# 1. Setup paths
# --------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GRAPHS_FILE = DATA_DIR / "train_graphs.pt"

# verify file exists
if not GRAPHS_FILE.exists():
    raise FileNotFoundError(f"{GRAPHS_FILE} not found. Run convert_smiles_to_graphs.py first.")

# --------------------------------------
# 2. Load graphs
# --------------------------------------
graphs = torch.load(GRAPHS_FILE)
print(f"Loaded {len(graphs)} graphs from {GRAPHS_FILE}")

# --------------------------------------
# 3. Compute statistics
# --------------------------------------
node_counts = [g.num_nodes for g in graphs]
edge_counts = [g.edge_index.size(1)//2 for g in graphs]

print("\nOverall graph size statistics:")
print(f"  Nodes:  min={min(node_counts)}, max={max(node_counts)}, avg={sum(node_counts)/len(node_counts):.2f}")
print(f"  Edges:  min={min(edge_counts)}, max={max(edge_counts)}, avg={sum(edge_counts)/len(edge_counts):.2f}")

# show top-5 largest graphs by node count
sorted_idx = sorted(range(len(graphs)), key=lambda i: node_counts[i])
print("\nFive smallest graphs (nodes):")
for i in sorted_idx[:5]:
    print(f"  idx={i}, nodes={node_counts[i]}, edges={edge_counts[i]}")

print("\nFive largest graphs (nodes):")
for i in sorted_idx[-5:]:
    print(f"  idx={i}, nodes={node_counts[i]}, edges={edge_counts[i]}")

# --------------------------------------
# 4. Optional: atom-type global distribution
# --------------------------------------
print("\nAggregating atom-type distribution across all graphs...")
from rdkit import Chem
atom_counter = Counter()
for g in graphs:
    mol = Chem.MolFromSmiles(g.smiles)
    atom_counter.update(atom.GetSymbol() for atom in mol.GetAtoms())
print("Global atom counts:", dict(atom_counter))
