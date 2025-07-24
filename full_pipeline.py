"""
full_pipeline.py - Unified trainer and predictor for the
NeurIPS-2025 Open-Polymer Challenge.

* Uses YAML as the single entry point for configuration (GraphGym style).
* Trains 5 independent GRIT models for Tg, FFV, Tc, Density, and Rg respectively.
* The loss function is determined by the YAML file; final evaluation uses Weighted-MAE.
* Only creates a new polymer_loader.py in GRIT/loader/, leaving other source code untouched.
"""

from __future__ import annotations
import argparse, sys, uuid, time, os, shutil, subprocess, random, importlib.util
from pathlib import Path
import logging
from typing import List, Dict
from functools import partial
from tqdm import tqdm
import io

# --- Start: Fix for matplotlib backend on headless servers ---
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
# --- End: Fix ---

# --- Path setup to find the GRIT module ---
# This must be done before importing from grit
GRIT_DIR = Path(__file__).resolve().parent / "GRIT"
sys.path.append(str(GRIT_DIR))

import yaml
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger

# --- GRIT Imports (carefully selected) ---
from torch_geometric.graphgym.config import cfg as gym_cfg, set_cfg, load_cfg, dump_cfg
from yacs.config import CfgNode # Correct import location for CfgNode
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler, OptimizerConfig
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_geometric.graphgym.utils.epoch import is_eval_epoch
from torch_geometric.graphgym.checkpoint import save_ckpt, load_ckpt
from torch_geometric import seed_everything
from grit.logger import create_logger
from grit.optimizer.extra_optimizers import ExtendedSchedulerConfig
from grit.train.custom_train import train_epoch
from grit.transform.transforms import pre_transform_in_memory
from grit.transform.posenc_stats import compute_posenc_stats

# ----------------------------- Basic Paths / Constants ----------------------------- #
ROOT        = Path(__file__).resolve().parent
DATA_ROOT   = ROOT / "data"
SUPP_DIR    = DATA_ROOT / "train_supplement"
GRAPH_DIR   = SUPP_DIR / "graphs"
RESULTS_DIR = ROOT / "results"
CONFIG_SAVE = RESULTS_DIR / "cfg_runs"

TARGETS     = ["Tg", "FFV", "Tc", "Density", "Rg"]  # Fixed order
MINMAX = {  # Official min/max values for scoring
    'Tg': [-148.0297376, 472.25],
    'FFV': [0.2269924, 0.77709707],
    'Tc': [0.0465, 0.524],
    'Density': [0.748691234, 1.840998909],
    'Rg': [9.7283551, 34.672905605],
}
NULL_FOR_SUB = -9999.          # Placeholder value required by the competition

RDLogger.DisableLog("rdApp.*")

# ----------------------- 0) Canonical SMILES Helper ----------------------- #
def canon(smiles: str | float) -> str | None:
    if not isinstance(smiles, str):
        return None
    m = Chem.MolFromSmiles(smiles.strip())
    return None if m is None else Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)

# --------------------- 1) Consolidate CSVs (main + 4 supplements) --------------------- #
ALIASES = {
    "tc_mean": "Tc", "thermal_conductivity": "Tc", "tc": "Tc",
    "glass_transition_temp": "Tg", "glass_transition": "Tg",
    "ffv": "FFV", "fractional_free_volume": "FFV",
    "density": "Density", "rho": "Density",
    "rg": "Rg", "radius_of_gyration": "Rg",
}

def _standard(df: pd.DataFrame) -> pd.DataFrame:
    if "SMILES" not in df and "smiles" in df:
        df = df.rename(columns={"smiles": "SMILES"})
    df = df.rename(columns={c: ALIASES.get(c.lower(), c) for c in df.columns})
    keep = ["SMILES"] + [c for c in TARGETS if c in df.columns]
    df = df[keep].copy()
    for t in TARGETS:
        df[t] = pd.to_numeric(df.get(t, np.nan), errors="coerce")
    df["SMILES"] = df["SMILES"].map(canon)
    return df.dropna(subset=["SMILES"])

def consolidate() -> pd.DataFrame:
    frames = [
        _standard(pd.read_csv(DATA_ROOT / "train.csv")),
        _standard(pd.read_csv(SUPP_DIR / "dataset1.csv")),
        _standard(pd.read_csv(SUPP_DIR / "dataset2.csv")),
        _standard(pd.read_csv(SUPP_DIR / "dataset3.csv")),
        _standard(pd.read_csv(SUPP_DIR / "dataset4.csv")),
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.groupby("SMILES")[TARGETS].mean().reset_index()

# ------------------ 2) smiles2graph (using GRIT helper) ------------------- #
spec = importlib.util.spec_from_file_location(
    "ps", GRIT_DIR / "grit" / "loader" / "dataset" / "peptides_structural.py")
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)
smiles2graph = ps.smiles2graph

# ------------------------ 3) Generate / Load Graph Cache --------------------------- #
def build_graph_cache() -> None:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    if (GRAPH_DIR / "train_graphs.pt").exists() and (GRAPH_DIR / "test_graphs.pt").exists():
        return
    print("Building graph cache...")
    df = consolidate()
    graphs: List[Data] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Processing training graphs"):
        g = smiles2graph(row.SMILES)
        y = torch.tensor([getattr(row, t) for t in TARGETS], dtype=torch.float)
        graphs.append(Data(
            x=torch.from_numpy(g["node_feat"]).long(),
            edge_index=torch.from_numpy(g["edge_index"]).long(),
            edge_attr=torch.from_numpy(g["edge_feat"]).long(),
            y=y,
            y_mask=~torch.isnan(y)
        ))
    torch.save(graphs, GRAPH_DIR/"train_graphs.pt")

    # Generate test graphs
    test_df = pd.read_csv(DATA_ROOT/"test.csv")
    if "smiles" in test_df: test_df = test_df.rename(columns={"smiles":"SMILES"})
    test_df["SMILES"] = test_df["SMILES"].map(canon)
    t_graphs = []
    for smi in tqdm(test_df.SMILES.dropna(), total=len(test_df.SMILES.dropna()), desc="Processing test graphs"):
        g = smiles2graph(smi)
        t_graphs.append(Data(
            x=torch.from_numpy(g["node_feat"]).long(),
            edge_index=torch.from_numpy(g["edge_index"]).long(),
            edge_attr=torch.from_numpy(g["edge_feat"]).long()))
    torch.save(t_graphs, GRAPH_DIR/"test_graphs.pt")
    print("Graph cache built successfully.")

# -------------- 4) Create a subset of graphs for a specific target -------------- #
def subset_graphs(full: List[Data], target_idx: int) -> List[Data]:
    out = []
    for d in full:
        if torch.isfinite(d.y[target_idx]):
            d2 = Data(**{k: v.clone() for k, v in d.__dict__.items() if k[0] != "_"})
            d2.y = d.y[target_idx:target_idx+1]
            d2.y_mask = torch.tensor([True])
            out.append(d2)
    return out

# ---------------------- 5) Weighted MAE Function (Refactored) ---------------------- #
def calculate_mae(truth, pred):
    """Calculates the simple Mean Absolute Error."""
    return np.mean(np.abs(truth - pred))

def weighted_mae(sol: pd.DataFrame, sub: pd.DataFrame) -> float:
    """Calculates the Weighted Mean Absolute Error exactly as defined by the competition formula."""
    eval_df = pd.merge(sol, sub.rename(columns={c: f'{c}_pred' for c in TARGETS}), on='SMILES', how='left')

    K = len(TARGETS)
    n_i = {col: (eval_df[col] != NULL_FOR_SUB).sum() for col in TARGETS}
    r_i = {col: MINMAX[col][1] - MINMAX[col][0] for col in TARGETS}
    sqrt_inv_n = {col: np.sqrt(1 / (n_i[col] + 1e-9)) for col in TARGETS}
    sum_sqrt_inv_n = sum(sqrt_inv_n.values())
    freq_weights = {col: (K * val / sum_sqrt_inv_n) for col, val in sqrt_inv_n.items()}

    total_error = 0
    for col in TARGETS:
        mask = eval_df[col] != NULL_FOR_SUB
        if mask.any():
            mae = calculate_mae(eval_df.loc[mask, col], eval_df.loc[mask, f'{col}_pred'])
            scale_norm = 1 / (r_i[col] + 1e-9)
            freq_w = freq_weights[col]
            total_error += freq_w * (mae * scale_norm)

    return float(total_error)

# --------------------- 6) Write polymer_loader.py --------------------- #
def ensure_poly_loader():
    loader_py = GRIT_DIR / "grit" / "loader" / "polymer_loader.py"
    loader_py.write_text(
        """
from pathlib import Path
import torch, random
from torch_geometric.data import InMemoryDataset

class PolymerDS(InMemoryDataset):
    def __init__(self, root, target_idx: int, **kw):
        super().__init__(root, **kw)
        graphs = torch.load(Path(root) / "train_supplement" / "graphs" / "train_graphs.pt", map_location='cpu', weights_only=False)
        graphs = [g for g in graphs if g.y_mask[target_idx]]
        for g in graphs:
            g.y = g.y[target_idx:target_idx+1]
            g.y_mask = torch.tensor([True])
        self.data, self.slices = self.collate(graphs)

    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return []
    def download(self): pass
    def process(self): pass
""")

# --- Helper functions from GRIT/main.py ---
def new_optimizer_config(cfg):
    return OptimizerConfig(optimizer=cfg.optim.optimizer, base_lr=cfg.optim.base_lr,
                           weight_decay=cfg.optim.weight_decay, momentum=cfg.optim.momentum)

def new_scheduler_config(cfg):
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler, steps=cfg.optim.steps, lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch, reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience, min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs, train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period, num_cycles=cfg.optim.num_cycles,
        min_lr_mode=cfg.optim.min_lr_mode)

def custom_set_run_dir(cfg, run_id):
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    # Do not delete existing directories, to allow resuming
    os.makedirs(cfg.run_dir, exist_ok=True)

@torch.no_grad()
def eval_epoch_with_progress(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in tqdm(loader, desc=f"Evaluating {split} set", leave=False):
        batch.split = split
        batch.to(torch.device(gym_cfg.device))
        pred, true = model(batch)
        loss, pred_score = compute_loss(pred, true)
        _true = true.detach().to('cpu', non_blocking=True)
        _pred = pred_score.detach().to('cpu', non_blocking=True)
        logger.update_stats(true=_true, pred=_pred, loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=gym_cfg.params, dataset_name=gym_cfg.dataset.name)
        time_start = time.time()

def plot_and_save_loss_curve(perf_history: List[List[Dict]], save_dir: Path):
    """Plots and saves the training and validation loss curves."""
    train_loss = [epoch['loss'] for epoch in perf_history[0]]
    val_loss = [epoch['loss'] for epoch in perf_history[1]]
    epochs = range(len(train_loss))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title(f'Loss Curve - {save_dir.name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = save_dir / "loss_curve.png"
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Loss curve saved to {save_path}")

# ---------------- 7) Single-Target Training Function (Refactored) ---------------- #
def run_training_process(device: str, PolymerDS_class: type) -> Path:
    """Main training routine for a single target."""
    run_id, seed = gym_cfg.seed, gym_cfg.seed
    custom_set_run_dir(gym_cfg, run_id)
    set_printing()
    gym_cfg.seed = seed
    gym_cfg.run_id = run_id
    seed_everything(gym_cfg.seed)
    gym_cfg.device = device

    dataset = PolymerDS_class(root=DATA_ROOT, target_idx=gym_cfg.dataset.target_idx)
    
    pe_enabled_list = []
    if gym_cfg.posenc_RRWP.enable:
        pe_enabled_list.append('RRWP')
    if pe_enabled_list:
        logging.info(f"Precomputing Positional Encoding statistics: {pe_enabled_list} for all graphs...")
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        pre_transform_in_memory(dataset, partial(compute_posenc_stats, pe_types=pe_enabled_list,
                                        is_undirected=is_undirected, cfg=gym_cfg), show_progress=True)

    _SPLITS = (0.8, 0.1, 0.1)
    N = len(dataset)
    ids = list(range(N))
    random.Random(42).shuffle(ids)
    n_tr = int(_SPLITS[0] * N)
    n_val = int(_SPLITS[1] * N)
    dataset._data['train_graph_index'] = torch.tensor(ids[:n_tr])
    dataset._data['val_graph_index'] = torch.tensor(ids[n_tr:n_tr + n_val])
    dataset._data['test_graph_index'] = torch.tensor(ids[n_tr + n_val:])

    train_loader = DataLoader(dataset[dataset._data['train_graph_index']], batch_size=gym_cfg.train.batch_size, shuffle=True, num_workers=gym_cfg.num_workers)
    val_loader = DataLoader(dataset[dataset._data['val_graph_index']], batch_size=gym_cfg.train.batch_size, shuffle=False, num_workers=gym_cfg.num_workers)
    test_loader = DataLoader(dataset[dataset._data['test_graph_index']], batch_size=gym_cfg.train.batch_size, shuffle=False, num_workers=gym_cfg.num_workers)
    
    gym_cfg.share.num_splits = 3

    loggers = create_logger()
    model = create_model().to(device)
    optimizer = create_optimizer(model.parameters(), new_optimizer_config(gym_cfg))
    scheduler = create_scheduler(optimizer, new_scheduler_config(gym_cfg))

    logging.info(model)
    gym_cfg.params = params_count(model)
    logging.info(f'Num parameters: {gym_cfg.params}')

    start_epoch = 0
    best_val_mae = float('inf')
    best_epoch = -1
    val_perf = {}
    perf_history = [[] for _ in range(gym_cfg.share.num_splits)]

    for cur_epoch in tqdm(range(start_epoch, gym_cfg.optim.max_epoch), desc=f"Training Target: {TARGETS[gym_cfg.dataset.target_idx]}"):
        train_epoch(loggers[0], train_loader, model, optimizer, scheduler, gym_cfg.optim.batch_accumulation)
        perf_history[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch):
            eval_epoch_with_progress(loggers[1], val_loader, model, split='val')
            val_perf = loggers[1].write_epoch(cur_epoch)
            perf_history[1].append(val_perf)
            eval_epoch_with_progress(loggers[2], test_loader, model, split='test')
            perf_history[2].append(loggers[2].write_epoch(cur_epoch))
            
            if val_perf['mae'] < best_val_mae:
                best_val_mae = val_perf['mae']
                best_epoch = cur_epoch
                logging.info(f"New best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
                save_ckpt(model, optimizer, scheduler, cur_epoch)
        
        if gym_cfg.optim.scheduler == 'reduce_on_plateau':
            if val_perf:
                scheduler.step(val_perf['loss'])
            else:
                scheduler.step()
        else:
            scheduler.step()

    logging.info(f"Training finished. Best validation MAE: {best_val_mae:.4f} at epoch {best_epoch}")
    
    plot_and_save_loss_curve(perf_history, Path(gym_cfg.run_dir))

    ckpt_path = Path(gym_cfg.run_dir) / "ckpt" / f"{best_epoch}.ckpt"
    if not ckpt_path.exists():
         raise RuntimeError(f"Best checkpoint not found at epoch {best_epoch} in {gym_cfg.run_dir}")
    
    final_ckpt_path = ckpt_path.parent / "ckpt_best.pth"
    shutil.copy(ckpt_path, final_ckpt_path)
    
    return final_ckpt_path

def train_one_target(cfg_template: Dict, target: str, idx: int, device: str, 
                     PolymerDS_class: type, num_node_types: int, num_edge_types: int, cfg_basename: str) -> Path:
    """Generate a config for a given target, train with GRIT, and return the checkpoint path."""
    out_dir = RESULTS_DIR / cfg_basename / target
    run_dir_contender = out_dir / str(cfg_template.get('seed', 0))
    final_ckpt_path = run_dir_contender / "ckpt" / "ckpt_best.pth"
    if final_ckpt_path.exists():
        logging.info(f"Found existing completed run for target '{target}'. Skipping training.")
        return final_ckpt_path

    out_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_dict = cfg_template.copy()
    cfg_dict['out_dir'] = str(out_dir)
    cfg_dict['dataset'] = cfg_dict.get('dataset', {})
    cfg_dict['dataset']['dir'] = str(DATA_ROOT)
    cfg_dict['dataset']['name'] = 'PolymerData'
    cfg_dict['dataset']['format'] = 'PyG-PolymerData'
    cfg_dict['dataset']['task_type'] = 'regression'
    cfg_dict['dataset']['target_idx'] = idx
    cfg_dict['dataset']['node_encoder_num_types'] = num_node_types
    cfg_dict['dataset']['edge_encoder_num_types'] = num_edge_types
    cfg_dict['train'] = cfg_dict.get('train', {})
    cfg_dict['train']['mode'] = 'custom'
    cfg_dict['metric_best'] = 'mae'
    if 'optim' not in cfg_dict: cfg_dict['optim'] = {}
    if 'max_epoch' not in cfg_dict['optim']: cfg_dict['optim']['max_epoch'] = 200

    cfg_path = out_dir / f"cfg_{target}.yaml"
    with open(cfg_path, 'w') as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)

    set_cfg(gym_cfg)
    gym_cfg.set_new_allowed(True)
    gym_cfg.work_dir = str(ROOT)
    args = argparse.Namespace(cfg_file=str(cfg_path), repeat=1, mark_done=False, opts=[])
    load_cfg(gym_cfg, args)
    dump_cfg(gym_cfg)

    return run_training_process(device, PolymerDS_class)

# --- Start: Wrapper for a list of Data objects to behave like a Dataset ---
class ListDataset(InMemoryDataset):
    def __init__(self, graphs: List[Data]):
        super().__init__()
        self.data, self.slices = self.collate(graphs)

    def __len__(self) -> int:
        if self.slices is None:
            return 0
        key = list(self.slices.keys())[0]
        return self.slices[key].size(0) - 1

    def get(self, idx: int) -> Data:
        return super().get(idx)
# --- End: Wrapper ---

# ----------------------- 8) Main Function ----------------------- #
def main():
    total_start_time = time.time()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, help='Path to base YAML config')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    with open(args.cfg) as f:
        base_cfg = yaml.safe_load(f)

    build_graph_cache()
    ensure_poly_loader()

    loader_py = GRIT_DIR / "grit" / "loader" / "polymer_loader.py"
    spec = importlib.util.spec_from_file_location("polymer_loader", loader_py)
    poly_loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(poly_loader_module)
    PolymerDS_class = poly_loader_module.PolymerDS

    graphs = torch.load(GRAPH_DIR / "train_graphs.pt", map_location='cpu', weights_only=False)
    max_node_type = 0
    max_edge_type = 0
    for g in graphs:
        max_node_type = max(max_node_type, g.x.max().item())
        max_edge_type = max(max_edge_type, g.edge_attr.max().item())
    num_node_types = max_node_type + 1
    num_edge_types = max_edge_type + 1
    logging.info(f"Calculated num_node_types: {num_node_types}")
    logging.info(f"Calculated num_edge_types: {num_edge_types}")

    train_df = consolidate()
    oof = train_df[['SMILES']].copy()
    test_df = pd.read_csv(DATA_ROOT/'test.csv')
    if 'smiles' in test_df: test_df = test_df.rename(columns={'smiles':'SMILES'})
    test_df["SMILES"] = test_df["SMILES"].map(canon)
    sub = test_df[['SMILES']].copy()

    cfg_basename = Path(args.cfg).stem
    ckpts = {}
    for idx, tgt in enumerate(TARGETS):
        print(f"\n==== Training target: {tgt} ====")
        ckpts[tgt] = train_one_target(base_cfg, tgt, idx, args.device, 
                                      PolymerDS_class, num_node_types, num_edge_types, cfg_basename)

    # --- Start: Fix for test set PE --- 
    t_graphs_list = torch.load(GRAPH_DIR/'test_graphs.pt', weights_only=False)
    pe_enabled_list = []
    if base_cfg.get('posenc_RRWP', {}).get('enable', False):
        pe_enabled_list.append('RRWP')
    if pe_enabled_list:
        logging.info(f"Precomputing Positional Encoding statistics for test set: {pe_enabled_list}...")
        temp_cfg = CfgNode()
        set_cfg(temp_cfg)
        temp_cfg.set_new_allowed(True)
        temp_cfg.merge_from_file(args.cfg)
        temp_cfg.dataset.node_encoder_num_types = num_node_types
        temp_cfg.dataset.edge_encoder_num_types = num_edge_types
        is_undirected = all(d.is_undirected() for d in t_graphs_list[:10])
        
        t_graphs = ListDataset(t_graphs_list)
        pre_transform_in_memory(t_graphs, partial(compute_posenc_stats, pe_types=pe_enabled_list,
                                        is_undirected=is_undirected, cfg=temp_cfg), show_progress=True)
    else:
        t_graphs = ListDataset(t_graphs_list)
    # --- End: Fix for test set PE ---

    for idx, tgt in enumerate(TARGETS):
        print(f"\n==== Predicting for target: {tgt} ====")
        set_cfg(gym_cfg)
        gym_cfg.set_new_allowed(True)
        cfg_path = (RESULTS_DIR / cfg_basename / tgt) / f"cfg_{tgt}.yaml"
        args_load = argparse.Namespace(cfg_file=str(cfg_path), repeat=1, mark_done=False, opts=[])
        load_cfg(gym_cfg, args_load)

        model = create_model().to(args.device)
        model_state = torch.load(ckpts[tgt], map_location=args.device)['model_state']
        model.load_state_dict(model_state)
        model.eval()

        dl = DataLoader(t_graphs, batch_size=128)
        preds=[]
        with torch.no_grad():
            for b in tqdm(dl, desc=f"Predicting {tgt} on test set", leave=False):
                pred, _ = model(b.to(args.device))
                preds.append(pred.cpu().view(-1))
        sub[tgt] = torch.cat(preds).numpy()

        full = torch.load(GRAPH_DIR/'train_graphs.pt', map_location='cpu', weights_only=False)
        subset = subset_graphs(full, idx)
        dl_oof = DataLoader(subset, batch_size=128)
        o=[]
        with torch.no_grad():
            for b in tqdm(dl_oof, desc=f"Predicting {tgt} for OOF", leave=False):
                pred, _ = model(b.to(args.device))
                o.append(pred.cpu().view(-1))
        oof[tgt] = torch.cat(o).numpy()

    sol = train_df[['SMILES']+TARGETS].copy()
    sol = sol.fillna(NULL_FOR_SUB)
    wmae = weighted_mae(sol, oof)
    print(f"\n=== Weighted MAE on training set: {wmae:.5f} ===")

    oof.to_csv(ROOT/'oof.csv', index=False)
    sub_out = pd.read_csv(DATA_ROOT/'sample_submission.csv')
    sub_out[TARGETS] = sub[TARGETS].values
    sub_out.to_csv(ROOT/'submission.csv', index=False)
    print("Saved oof.csv & submission.csv")

    total_end_time = time.time()
    elapsed_seconds = total_end_time - total_start_time
    minutes, seconds = divmod(elapsed_seconds, 60)
    print(f"\nTotal execution time: {int(minutes)} minutes and {seconds:.2f} seconds.")

if __name__ == "__main__":
    main()