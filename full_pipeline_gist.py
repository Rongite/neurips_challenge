"""
full_pipeline_gist.py - Unified trainer and predictor using GIST architecture for the
NeurIPS-2025 Open-Polymer Challenge.

* Uses YAML as the single entry point for configuration (GraphGym style).
* Trains 5 independent GIST models for Tg, FFV, Tc, Density, and Rg respectively.
* Two-stage training: Stage 1 (8:1:1 split) + Stage 2 (full data fine-tuning).
* Comprehensive model evaluation and CSV reporting system.
* Complete checkpoint resume capability.
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

# --- Path setup to find the GIST module ---
# This must be done before importing from gist
GIST_DIR = Path(__file__).resolve().parent / "GIST"
sys.path.append(str(GIST_DIR))

import yaml
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from rdkit import Chem, RDLogger

# --- GIST Imports (adapted from GRIT) ---
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

# ----------------------- 0) Data Range Statistics and Model Evaluation ----------------------- #
def calculate_data_ranges() -> Dict[str, Dict[str, float]]:
    """Calculate actual min/max ranges from all training datasets."""
    data_files = [
        DATA_ROOT / "train.csv",
        SUPP_DIR / "dataset1.csv",
        SUPP_DIR / "dataset2.csv", 
        SUPP_DIR / "dataset3.csv",
        SUPP_DIR / "dataset4.csv"
    ]
    
    ranges = {target: {'min': float('inf'), 'max': float('-inf'), 'count': 0} for target in TARGETS}
    
    for file_path in data_files:
        if file_path.exists():
            df = pd.read_csv(file_path)
            df = _standard(df)  # Use existing standardization function
            
            for target in TARGETS:
                if target in df.columns:
                    values = df[target].dropna()
                    if len(values) > 0:
                        ranges[target]['min'] = min(ranges[target]['min'], values.min())
                        ranges[target]['max'] = max(ranges[target]['max'], values.max())
                        ranges[target]['count'] += len(values)
    
    # Convert to final format
    result = {}
    for target in TARGETS:
        if ranges[target]['count'] > 0:
            range_val = ranges[target]['max'] - ranges[target]['min']
            result[target] = {
                'min': ranges[target]['min'],
                'max': ranges[target]['max'], 
                'range': range_val,
                'count': ranges[target]['count']
            }
        else:
            result[target] = {'min': 0, 'max': 1, 'range': 1, 'count': 0}
    
    return result

def evaluate_model_performance(mae: float, target: str, data_ranges: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """Evaluate model performance based on MAE and data range."""
    if target not in data_ranges or data_ranges[target]['range'] == 0:
        return {'grade': 'Unknown', 'level': 'Cannot evaluate', 'relative_error': 'N/A'}
    
    range_val = data_ranges[target]['range']
    relative_error = (mae / range_val) * 100
    
    # Define evaluation criteria
    if relative_error < 5:
        grade = 'A+'
        level = 'Excellent'
    elif relative_error < 8:
        grade = 'A'
        level = 'Very Good'
    elif relative_error < 12:
        grade = 'B+'
        level = 'Good'
    elif relative_error < 18:
        grade = 'B'
        level = 'Fair'
    elif relative_error < 25:
        grade = 'C'
        level = 'Acceptable'
    else:
        grade = 'D'
        level = 'Poor'
    
    return {
        'grade': grade,
        'level': level,
        'relative_error': f'{relative_error:.2f}%'
    }

def save_stage1_evaluation_report(stage1_results: Dict[str, Dict], data_ranges: Dict[str, Dict[str, float]], save_dir: Path):
    """Save Stage 1 evaluation results to CSV file."""
    report_data = []
    
    for target in TARGETS:
        if target in stage1_results:
            result = stage1_results[target]
            ranges = data_ranges[target]
            
            report_data.append({
                'Target': target,
                'MAE': f"{result['mae']:.4f}",
                'Data_Range': f"{ranges['range']:.4f}",
                'Data_Min': f"{ranges['min']:.4f}",
                'Data_Max': f"{ranges['max']:.4f}",
                'Sample_Count': ranges['count'],
                'Relative_Error': result['evaluation']['relative_error'],
                'Performance_Grade': result['evaluation']['grade'],
                'Performance_Level': result['evaluation']['level'],
                'Training_Status': 'Completed'
            })
        else:
            # Handle missing targets
            ranges = data_ranges.get(target, {'range': 0, 'min': 0, 'max': 0, 'count': 0})
            report_data.append({
                'Target': target,
                'MAE': 'N/A',
                'Data_Range': f"{ranges['range']:.4f}",
                'Data_Min': f"{ranges['min']:.4f}",
                'Data_Max': f"{ranges['max']:.4f}",
                'Sample_Count': ranges['count'],
                'Relative_Error': 'N/A',
                'Performance_Grade': 'N/A',
                'Performance_Level': 'Not Trained',
                'Training_Status': 'Pending'
            })
    
    # Save to CSV
    report_df = pd.DataFrame(report_data)
    report_path = save_dir / "stage1_evaluation_report_gist.csv"
    report_df.to_csv(report_path, index=False)
    
    logging.info(f"Stage 1 evaluation report saved to: {report_path}")
    return report_path

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

# ------------------ 2) smiles2graph (using GIST helper) ------------------- #
spec = importlib.util.spec_from_file_location(
    "ps", GIST_DIR / "grit" / "loader" / "dataset" / "peptides_structural.py")
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

# --------------------- 6) Write polymer loader for GIST --------------------- #
def ensure_poly_loader():
    loader_py = GIST_DIR / "grit" / "loader" / "polymer_loader.py"
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

# --- Helper functions from GIST/main.py ---
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

def plot_stage2_loss_curve(train_losses: List[float], save_dir: Path, target_name: str):
    """Plots and saves the Stage 2 training loss curve."""
    epochs = range(len(train_losses))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Stage 2 Training Loss', color='green', linewidth=2)
    plt.title(f'Stage 2 Fine-tuning Loss Curve - {target_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = save_dir / "stage2_loss_curve.png"
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Stage 2 loss curve saved to {save_path}")

# ---------------- 7) Single-Target Training Functions (Two-Stage) ---------------- #
def run_stage1_training(device: str, PolymerDS_class: type) -> tuple[Path, float]:
    """Stage 1: Train with 8:1:1 split, return best checkpoint and test wMAE."""
    run_id, seed = gym_cfg.seed, gym_cfg.seed
    custom_set_run_dir(gym_cfg, run_id)
    
    # Check if Stage 1 is already completed
    final_ckpt_path = Path(gym_cfg.run_dir) / "ckpt" / "ckpt_best.pth"
    mae_file = Path(gym_cfg.run_dir) / "stage1_test_mae.txt"
    
    if final_ckpt_path.exists() and mae_file.exists():
        logging.info("✅ Stage 1 already completed. Loading existing results.")
        try:
            with open(mae_file, 'r') as f:
                test_mae = float(f.read().strip())
            return final_ckpt_path, test_mae
        except:
            logging.warning("Failed to load existing test MAE. Re-calculating...")
    elif final_ckpt_path.exists():
        logging.info("🔄 Found Stage 1 checkpoint but missing test MAE. Re-calculating test MAE...")
        # Load model and calculate test MAE on existing checkpoint
        dataset = PolymerDS_class(root=DATA_ROOT, target_idx=gym_cfg.dataset.target_idx)
        
        # Apply PE if needed
        pe_enabled_list = []
        if gym_cfg.posenc_RRWP.enable:
            pe_enabled_list.append('RRWP')
        if pe_enabled_list:
            is_undirected = all(d.is_undirected() for d in dataset[:10])
            pre_transform_in_memory(dataset, partial(compute_posenc_stats, pe_types=pe_enabled_list,
                                            is_undirected=is_undirected, cfg=gym_cfg), show_progress=True)
        
        # Recreate splits
        _SPLITS = (0.8, 0.1, 0.1)
        N = len(dataset)
        ids = list(range(N))
        random.Random(42).shuffle(ids)
        n_tr = int(_SPLITS[0] * N)
        n_val = int(_SPLITS[1] * N)
        dataset._data['test_graph_index'] = torch.tensor(ids[n_tr + n_val:])
        
        # Calculate test MAE
        test_graphs = [dataset.get(i) for i in dataset._data['test_graph_index']]
        test_loader_for_eval = DataLoader(test_graphs, batch_size=128, shuffle=False)
        
        model_for_eval = create_model().to(device)
        model_state = torch.load(final_ckpt_path, map_location=device)['model_state']
        model_for_eval.load_state_dict(model_state)
        model_for_eval.eval()
        
        test_preds = []
        test_trues = []
        with torch.no_grad():
            for batch in test_loader_for_eval:
                batch = batch.to(device)
                pred, true = model_for_eval(batch)
                test_preds.append(pred.cpu().view(-1))
                test_trues.append(true.cpu().view(-1))
        
        test_preds = torch.cat(test_preds).numpy()
        test_trues = torch.cat(test_trues).numpy()
        test_mae = np.mean(np.abs(test_preds - test_trues))
        
        # Save test MAE
        with open(mae_file, 'w') as f:
            f.write(str(test_mae))
        
        logging.info(f"Calculated Stage 1 Test MAE for target {TARGETS[gym_cfg.dataset.target_idx]}: {test_mae:.4f}")
        return final_ckpt_path, test_mae
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
    
    # Calculate wMAE on held-out test set
    test_graphs = [dataset.get(i) for i in dataset._data['test_graph_index']]
    test_loader_for_eval = DataLoader(test_graphs, batch_size=128, shuffle=False)
    
    # Load best model for evaluation
    model_for_eval = create_model().to(device)
    model_state = torch.load(final_ckpt_path, map_location=device)['model_state']
    model_for_eval.load_state_dict(model_state)
    model_for_eval.eval()
    
    # Get predictions on test set
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for batch in test_loader_for_eval:
            batch = batch.to(device)
            pred, true = model_for_eval(batch)
            test_preds.append(pred.cpu().view(-1))
            test_trues.append(true.cpu().view(-1))
    
    test_preds = torch.cat(test_preds).numpy()
    test_trues = torch.cat(test_trues).numpy()
    
    # Calculate simple MAE for this target (will be used in wMAE calculation later)
    test_mae = np.mean(np.abs(test_preds - test_trues))
    target_name = TARGETS[gym_cfg.dataset.target_idx]
    
    logging.info(f"Stage 1 Test MAE for target {target_name}: {test_mae:.4f}")
    
    # Save test MAE to file for resume capability
    mae_file = Path(gym_cfg.run_dir) / "stage1_test_mae.txt"
    with open(mae_file, 'w') as f:
        f.write(str(test_mae))
    
    return final_ckpt_path, test_mae

def run_stage2_training(device: str, PolymerDS_class: type, stage1_ckpt: Path) -> Path:
    """Stage 2: Fine-tune on full dataset using stage1 checkpoint as initialization."""
    
    # Check if Stage 2 is already completed
    stage2_ckpt_path = Path(gym_cfg.run_dir) / "ckpt" / "stage2" / "ckpt_final.pth"
    if stage2_ckpt_path.exists():
        logging.info("✅ Stage 2 already completed. Loading existing checkpoint.")
        return stage2_ckpt_path
    
    # Create dataset with all training data (no splits)
    dataset = PolymerDS_class(root=DATA_ROOT, target_idx=gym_cfg.dataset.target_idx)
    
    # Apply positional encoding if needed
    pe_enabled_list = []
    if gym_cfg.posenc_RRWP.enable:
        pe_enabled_list.append('RRWP')
    if pe_enabled_list:
        logging.info(f"Precomputing Positional Encoding for Stage 2: {pe_enabled_list}")
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        pre_transform_in_memory(dataset, partial(compute_posenc_stats, pe_types=pe_enabled_list,
                                        is_undirected=is_undirected, cfg=gym_cfg), show_progress=True)
    
    # Use all data for training (no validation set in stage 2)
    train_loader = DataLoader(dataset, batch_size=gym_cfg.train.batch_size, shuffle=True, num_workers=gym_cfg.num_workers)
    
    # Create model and load stage1 checkpoint
    model = create_model().to(device)
    stage1_state = torch.load(stage1_ckpt, map_location=device)
    model.load_state_dict(stage1_state['model_state'])
    
    # Use reduced learning rate for fine-tuning (1/10 of original)
    original_lr = gym_cfg.optim.base_lr
    stage2_lr = original_lr * 0.1
    
    # Use fewer epochs (1/3 of original)
    stage2_epochs = max(10, gym_cfg.optim.max_epoch // 3)
    logging.info(f"Stage 2: Using LR={stage2_lr:.6f} (vs Stage 1: {original_lr:.6f}), Epochs={stage2_epochs} (vs Stage 1: {gym_cfg.optim.max_epoch})")
    
    # Create optimizer config with reduced learning rate
    stage2_optim_cfg = new_optimizer_config(gym_cfg)
    stage2_optim_cfg.base_lr = stage2_lr
    
    optimizer = create_optimizer(model.parameters(), stage2_optim_cfg)
    scheduler = create_scheduler(optimizer, new_scheduler_config(gym_cfg))
    
    # Create logger for stage 2
    loggers = create_logger()
    
    logging.info(f"Stage 2 training for target {TARGETS[gym_cfg.dataset.target_idx]}")
    
    # Record loss for plotting
    stage2_losses = []
    
    for cur_epoch in tqdm(range(stage2_epochs), desc=f"Stage 2 - Target: {TARGETS[gym_cfg.dataset.target_idx]}"):
        train_epoch(loggers[0], train_loader, model, optimizer, scheduler, gym_cfg.optim.batch_accumulation)
        
        # Record epoch performance
        epoch_perf = loggers[0].write_epoch(cur_epoch)
        stage2_losses.append(epoch_perf['loss'])
        
        # Simple scheduler step (no validation-based scheduling in stage 2)
        if gym_cfg.optim.scheduler != 'reduce_on_plateau':
            scheduler.step()
    
    # Save final stage 2 checkpoint
    stage2_ckpt_dir = Path(gym_cfg.run_dir) / "ckpt" / "stage2"
    stage2_ckpt_dir.mkdir(parents=True, exist_ok=True)
    stage2_ckpt_path = stage2_ckpt_dir / "ckpt_final.pth"
    
    torch.save({
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch': stage2_epochs
    }, stage2_ckpt_path)
    
    # Plot Stage 2 loss curve
    plot_stage2_loss_curve(stage2_losses, Path(gym_cfg.run_dir), TARGETS[gym_cfg.dataset.target_idx])
    
    logging.info(f"Stage 2 training completed. Final checkpoint saved to {stage2_ckpt_path}")
    return stage2_ckpt_path

def check_training_status(out_dir: Path, target: str, cfg_template: Dict) -> dict:
    """Check the current training status and return resumption info."""
    run_dir = out_dir / str(cfg_template.get('seed', 0))
    stage1_ckpt_path = run_dir / "ckpt" / "ckpt_best.pth"
    stage2_ckpt_path = run_dir / "ckpt" / "stage2" / "ckpt_final.pth"
    stage1_test_mae_file = run_dir / "stage1_test_mae.txt"
    
    # Check existing checkpoints and intermediate files
    stage1_complete = stage1_ckpt_path.exists()
    stage2_complete = stage2_ckpt_path.exists()
    
    # Try to load saved test MAE if available
    stage1_test_mae = 0.0
    if stage1_test_mae_file.exists():
        try:
            with open(stage1_test_mae_file, 'r') as f:
                stage1_test_mae = float(f.read().strip())
        except:
            stage1_test_mae = 0.0
    
    return {
        'run_dir': run_dir,
        'stage1_ckpt_path': stage1_ckpt_path,
        'stage2_ckpt_path': stage2_ckpt_path,
        'stage1_test_mae_file': stage1_test_mae_file,
        'stage1_complete': stage1_complete,
        'stage2_complete': stage2_complete,
        'stage1_test_mae': stage1_test_mae,
        'target': target
    }

def train_one_target(cfg_template: Dict, target: str, idx: int, device: str, 
                     PolymerDS_class: type, num_node_types: int, num_edge_types: int, cfg_basename: str) -> tuple[Path, Path, float]:
    """Two-stage training with comprehensive checkpoint resume support."""
    out_dir = RESULTS_DIR / cfg_basename / target
    
    # Check current training status
    status = check_training_status(out_dir, target, cfg_template)
    
    if status['stage1_complete'] and status['stage2_complete']:
        logging.info(f"✅ Found completed two-stage training for '{target}'. Skipping all training.")
        return status['stage1_ckpt_path'], status['stage2_ckpt_path'], status['stage1_test_mae']
    
    elif status['stage1_complete'] and not status['stage2_complete']:
        logging.info(f"🔄 Found completed Stage 1 for '{target}'. Skipping Stage 1, resuming from Stage 2.")
        # Load existing stage 1 results
        stage1_ckpt = status['stage1_ckpt_path']
        stage1_test_mae = status['stage1_test_mae']
        
        # Configure for Stage 2 only
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
        
        # Run Stage 2 only
        logging.info(f"Starting Stage 2 training for target {target}")
        stage2_ckpt = run_stage2_training(device, PolymerDS_class, stage1_ckpt)
        
        return stage1_ckpt, stage2_ckpt, stage1_test_mae
    
    else:
        logging.info(f"🆕 Starting fresh two-stage training for '{target}'.")

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

    # Stage 1: Train with 8:1:1 split
    logging.info(f"Starting Stage 1 training for target {target}")
    stage1_ckpt, stage1_test_mae = run_stage1_training(device, PolymerDS_class)
    
    # Stage 2: Fine-tune on full dataset
    logging.info(f"Starting Stage 2 training for target {target}")
    stage2_ckpt = run_stage2_training(device, PolymerDS_class, stage1_ckpt)
    
    return stage1_ckpt, stage2_ckpt, stage1_test_mae

def calculate_stage1_wmae(stage1_results: Dict[str, float]) -> float:
    """Calculate wMAE from Stage 1 test results across all targets."""
    # Create a dummy dataframe structure for wMAE calculation
    eval_data = []
    for target in TARGETS:
        if target in stage1_results:
            # Create dummy true/pred values with the exact MAE difference
            mae_val = stage1_results[target]
            eval_data.append({
                'SMILES': f'dummy_{target}',
                target: 1.0,  # dummy true value
                f'{target}_pred': 1.0 + mae_val  # pred = true + mae to get exact mae
            })
    
    if not eval_data:
        return float('inf')
    
    eval_df = pd.DataFrame(eval_data)
    
    # Calculate wMAE using existing function logic
    K = len(TARGETS)
    n_i = {col: 1 for col in TARGETS if col in stage1_results}  # Each target has 1 sample
    r_i = {col: MINMAX[col][1] - MINMAX[col][0] for col in TARGETS}
    sqrt_inv_n = {col: np.sqrt(1 / (n_i.get(col, 1) + 1e-9)) for col in TARGETS}
    sum_sqrt_inv_n = sum(sqrt_inv_n.values())
    freq_weights = {col: (K * val / sum_sqrt_inv_n) for col, val in sqrt_inv_n.items()}
    
    total_error = 0
    for col in TARGETS:
        if col in stage1_results:
            mae = stage1_results[col]
            scale_norm = 1 / (r_i[col] + 1e-9)
            freq_w = freq_weights[col]
            total_error += freq_w * (mae * scale_norm)
    
    return float(total_error)

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

    loader_py = GIST_DIR / "grit" / "loader" / "polymer_loader.py"
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
    test_df = pd.read_csv(DATA_ROOT/'test.csv')
    if 'smiles' in test_df: test_df = test_df.rename(columns={'smiles':'SMILES'})
    test_df["SMILES"] = test_df["SMILES"].map(canon)
    sub = test_df[['SMILES']].copy()

    cfg_basename = Path(args.cfg).stem
    
    # ============ STAGE 1: Training with 8:1:1 Split ============ #
    print("\n" + "="*60)
    print("STAGE 1: Training with 8:1:1 data split (GIST Architecture)")
    print("="*60)
    
    # Calculate data ranges for evaluation
    print("📊 Calculating data ranges for model evaluation...")
    data_ranges = calculate_data_ranges()
    print("Data range statistics:")
    for target, stats in data_ranges.items():
        print(f"  {target}: Range={stats['range']:.4f}, Count={stats['count']}")
    
    stage1_ckpts = {}
    stage2_ckpts = {}
    stage1_results = {}  # Store detailed results including evaluation
    
    for idx, tgt in enumerate(TARGETS):
        print(f"\n==== Stage 1+2 Training for target: {tgt} ====")
        
        # Quick status check before training
        out_dir = RESULTS_DIR / cfg_basename / tgt
        status = check_training_status(out_dir, tgt, base_cfg)
        
        if status['stage1_complete'] and status['stage2_complete']:
            print(f"✅ Target {tgt}: Both stages completed")
        elif status['stage1_complete'] and not status['stage2_complete']:
            print(f"🔄 Target {tgt}: Stage 1 done, will resume Stage 2")
        else:
            print(f"🆕 Target {tgt}: Starting fresh training")
        
        stage1_ckpt, stage2_ckpt, test_mae = train_one_target(
            base_cfg, tgt, idx, args.device, PolymerDS_class, 
            num_node_types, num_edge_types, cfg_basename
        )
        stage1_ckpts[tgt] = stage1_ckpt
        stage2_ckpts[tgt] = stage2_ckpt
        
        # Evaluate model performance
        evaluation = evaluate_model_performance(test_mae, tgt, data_ranges)
        stage1_results[tgt] = {
            'mae': test_mae,
            'evaluation': evaluation
        }
        
        # Display evaluation results
        print(f"📈 {tgt} Model Evaluation:")
        print(f"   MAE: {test_mae:.4f}")
        print(f"   Relative Error: {evaluation['relative_error']}")
        print(f"   Performance Grade: {evaluation['grade']} ({evaluation['level']})")
        
        # Also keep the old format for compatibility
        stage1_test_maes = {k: v['mae'] for k, v in stage1_results.items()}

    # Generate and save Stage 1 evaluation report
    report_path = save_stage1_evaluation_report(stage1_results, data_ranges, RESULTS_DIR)
    
    # Calculate and report Stage 1 wMAE  
    stage1_wmae = calculate_stage1_wmae(stage1_test_maes)
    print(f"\n" + "="*60)
    print(f"STAGE 1 EVALUATION SUMMARY (GIST):")
    print(f"Individual Test MAEs: {stage1_test_maes}")
    print(f"Stage 1 Weighted MAE: {stage1_wmae:.6f}")
    print("\n📊 Model Performance Grades:")
    for target, result in stage1_results.items():
        eval_info = result['evaluation']
        print(f"   {target}: {eval_info['grade']} ({eval_info['level']}) - {eval_info['relative_error']} error")
    print(f"\n📋 Detailed evaluation report saved to: {report_path}")
    print("="*60)

    # Load test graphs for final predictions
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

    # ============ STAGE 2: Final Predictions with Stage 2 Models ============ #
    print(f"\n" + "="*60)
    print("STAGE 2: Generating final predictions with GIST models")
    print("="*60)
    
    for idx, tgt in enumerate(TARGETS):
        print(f"\n==== Predicting for target: {tgt} ====")
        set_cfg(gym_cfg)
        gym_cfg.set_new_allowed(True)
        cfg_path = (RESULTS_DIR / cfg_basename / tgt) / f"cfg_{tgt}.yaml"
        args_load = argparse.Namespace(cfg_file=str(cfg_path), repeat=1, mark_done=False, opts=[])
        load_cfg(gym_cfg, args_load)

        # Use Stage 2 (final) checkpoint for predictions
        model = create_model().to(args.device)
        model_state = torch.load(stage2_ckpts[tgt], map_location=args.device)['model_state']
        model.load_state_dict(model_state)
        model.eval()

        dl = DataLoader(t_graphs, batch_size=128)
        preds=[]
        with torch.no_grad():
            for b in tqdm(dl, desc=f"Predicting {tgt} on test set (Stage 2 GIST model)", leave=False):
                pred, _ = model(b.to(args.device))
                preds.append(pred.cpu().view(-1))
        sub[tgt] = torch.cat(preds).numpy()

    print(f"\n" + "="*60)
    print(f"FINAL RESULTS SUMMARY (GIST Architecture):")
    print(f"Stage 1 Individual Test MAEs: {stage1_test_maes}")
    print(f"Stage 1 Weighted MAE: {stage1_wmae:.6f}")
    print("\n📊 Final Model Performance Grades:")
    for target, result in stage1_results.items():
        eval_info = result['evaluation']
        print(f"   {target}: {eval_info['grade']} ({eval_info['level']}) - {eval_info['relative_error']} error")
    print(f"Stage 2 GIST models used for final submission")
    print(f"📋 Evaluation report: {report_path}")
    print("="*60)
    print(f"\n=== Test predictions completed successfully for all targets ===")
    print(f"Generated submission file: submission_gist.csv")
    sub_out = pd.read_csv(DATA_ROOT/'sample_submission.csv')
    sub_out[TARGETS] = sub[TARGETS].values
    sub_out.to_csv(ROOT/'submission_gist.csv', index=False)
    print("Saved submission_gist.csv (using Stage 2 GIST model predictions)")

    total_end_time = time.time()
    elapsed_seconds = total_end_time - total_start_time
    minutes, seconds = divmod(elapsed_seconds, 60)
    print(f"\nTotal execution time: {int(minutes)} minutes and {seconds:.2f} seconds.")

if __name__ == "__main__":
    main()