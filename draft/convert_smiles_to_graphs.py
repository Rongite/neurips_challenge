"""
Convert SMILES in data/train.csv → PyG Data objects using OGB smiles2graph from
GRIT/grit/loader/dataset/peptides_structural.py, then save all graphs and export 2 samples.
"""

import sys
import importlib.util
from pathlib import Path
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from torch_geometric.data import Data
from collections import Counter

# --------------------------------------
# 1) Paths setup
# --------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
TRAIN_CSV  = DATA_DIR / "train.csv"
OUT_FILE   = DATA_DIR / "train_graphs.pt"
SAMPLE_DIR = DATA_DIR / "samples"

# ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"{TRAIN_CSV} not found. Please place your Kaggle train.csv under data/")

# --------------------------------------
# 2) Silence RDKit warnings
# --------------------------------------
RDLogger.DisableLog("rdApp.*")

# --------------------------------------
# 3) Dynamically load smiles2graph from peptides_structural.py
# --------------------------------------
ps_path = BASE_DIR / "GRIT" / "grit" / "loader" / "dataset" / "peptides_structural.py"
if not ps_path.exists():
    raise FileNotFoundError(f"Cannot find {ps_path}; ensure you cloned the GRIT repo here")

spec   = importlib.util.spec_from_file_location("ps_mod", str(ps_path))
ps_mod = importlib.util.module_from_spec(spec)
sys.modules["ps_mod"] = ps_mod
spec.loader.exec_module(ps_mod)

# the OGB helper
smiles2graph = ps_mod.smiles2graph

# --------------------------------------
# 4) Read SMILES column
# --------------------------------------
df = pd.read_csv(TRAIN_CSV)
print("train.csv columns:", df.columns.tolist())

if "smiles" in df.columns:
    col = "smiles"
elif "SMILES" in df.columns:
    col = "SMILES"
else:
    col = df.columns[0]
    print(f"[WARN] using first column '{col}' as SMILES")

smiles_list = df[col].astype(str).tolist()
print(f"Total SMILES to process: {len(smiles_list)}")

# --------------------------------------
# 5) Convert loop with robust unpacking
# --------------------------------------
graphs = []
fails  = 0

for idx, smi in enumerate(smiles_list):
    # skip unparseable SMILES
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"[SKIP] idx={idx}: RDKit failed to parse {smi}")
        fails += 1
        continue

    # call the OGB helper
    try:
        graph = smiles2graph(smi)
    except Exception as e:
        print(f"[ERROR] idx={idx}: smiles2graph exception: {e!r}")
        fails += 1
        continue

    # verify dict keys
    required = {"edge_index", "edge_feat", "node_feat"}
    if not required.issubset(graph.keys()):
        print(f"[ERROR] idx={idx}: missing keys {required - set(graph.keys())}")
        fails += 1
        continue

    # build PyG Data
    try:
        data = Data(
            x          = torch.from_numpy(graph["node_feat"]).long(),
            edge_index = torch.from_numpy(graph["edge_index"]).long(),
            edge_attr  = torch.from_numpy(graph["edge_feat"]).long(),
        )
    except Exception as e:
        print(f"[ERROR] idx={idx}: failed to build Data: {e!r}")
        fails += 1
        continue

    data.smiles = smi
    graphs.append(data)

print(f"\nConversion done: {len(graphs)} succeeded, {fails} skipped/failed")

# --------------------------------------
# 6) Save resulting graphs
# --------------------------------------
torch.save(graphs, OUT_FILE)
print(f"Saved {len(graphs)} graphs to {OUT_FILE}")

# --------------------------------------
# 7) Output two sample stats & images
# --------------------------------------
for i in range(min(2, len(graphs))):
    g   = graphs[i]
    mol = Chem.MolFromSmiles(g.smiles)
    print(f"\nSample {i}: SMILES={g.smiles}")
    print("  Nodes:", g.num_nodes, "Edges:", g.edge_index.size(1)//2)
    atom_dist = Counter(a.GetSymbol() for a in mol.GetAtoms())
    bond_dist = Counter(str(b.GetBondType()) for b in mol.GetBonds())
    print("  Atom dist:", dict(atom_dist))
    print("  Bond dist:", dict(bond_dist))
    img_path = SAMPLE_DIR / f"sample_{i}.png"
    img = Draw.MolToImage(mol, size=(300, 300))
    img.save(img_path)
    print("  Image saved to", img_path.relative_to(BASE_DIR))