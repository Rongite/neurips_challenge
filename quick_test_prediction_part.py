import torch
import os
import sys
import shutil
import yaml
import argparse
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
import logging

# --- Setup paths and import from full_pipeline ---
# This allows us to reuse functions and variables without duplicating code
try:
    from full_pipeline import (
        ROOT, DATA_ROOT, GRAPH_DIR, RESULTS_DIR, GRIT_DIR, TARGETS, NULL_FOR_SUB,
        build_graph_cache, consolidate, subset_graphs, weighted_mae
    )
except ImportError:
    print("Error: Could not import from full_pipeline.py.")
    print("Please ensure this test script is in the same directory as full_pipeline.py")
    sys.exit(1)

# --- GRIT Imports ---
sys.path.append(str(GRIT_DIR))
from torch_geometric.graphgym.config import cfg as C, set_cfg, load_cfg, dump_cfg
from torch_geometric.graphgym.model_builder import create_model
from torch_geometric.graphgym.checkpoint import save_ckpt

def create_fake_checkpoint(target_name: str, base_cfg_path: str, num_node_types: int, num_edge_types: int):
    """Creates a fake but structurally valid checkpoint and config file."""
    fake_run_dir = RESULTS_DIR / f"fake_run_{target_name}"
    fake_run_id_dir = fake_run_dir / "0" # Corresponds to the default run_id
    fake_ckpt_dir = fake_run_id_dir / "ckpt"
    fake_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- Create a structurally valid, but randomly initialized, model state ---
    # 1. Load the config to define the model structure
    set_cfg(C)
    C.set_new_allowed(True)
    with open(base_cfg_path) as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict['dataset']['node_encoder_num_types'] = num_node_types
    cfg_dict['dataset']['edge_encoder_num_types'] = num_edge_types
    cfg_dict['out_dir'] = str(fake_run_dir)

    temp_cfg_path = fake_run_dir / "temp_cfg.yaml"
    with open(temp_cfg_path, 'w') as f:
        yaml.safe_dump(cfg_dict, f)
    args_load = argparse.Namespace(cfg_file=str(temp_cfg_path), repeat=1, mark_done=False, opts=[])
    load_cfg(C, args_load)
    
    # 2. Create a model instance based on this config
    model = create_model()
    
    # 3. Save this model's state using the official save_ckpt function
    original_run_dir = C.run_dir
    C.run_dir = str(fake_run_id_dir) # Temporarily set the global run_dir
    save_ckpt(model, None, None, epoch=1)
    C.run_dir = original_run_dir # Restore the global config

    fake_ckpt_path = fake_ckpt_dir / "1.ckpt"

    # Save the final config used for this fake run
    final_cfg_path = fake_run_dir / f"cfg_{target_name}.yaml"
    shutil.copy(temp_cfg_path, final_cfg_path)
    os.remove(temp_cfg_path)
    
    print(f"Created fake checkpoint for '{target_name}' at: {fake_ckpt_path}")
    print(f"Created fake config for '{target_name}' at: {final_cfg_path}")
    return fake_ckpt_path

def main():
    base_cfg_path = Path(__file__).resolve().parent / "configs" / "polymer-GRIT-RRWP.yaml"
    if not base_cfg_path.exists():
        print(f"Error: Base config not found at {base_cfg_path}")
        sys.exit(1)

    print("--- Step 1: Ensuring graph cache exists... ---")
    build_graph_cache()
    print("Graph cache is ready.")

    print("\n--- Step 2: Calculating feature type counts... ---")
    graphs = torch.load(GRAPH_DIR / "train_graphs.pt", map_location='cpu', weights_only=False)
    max_node_type = max(g.x.max().item() for g in graphs)
    max_edge_type = max(g.edge_attr.max().item() for g in graphs)
    num_node_types = max_node_type + 1
    num_edge_types = max_edge_type + 1
    print(f"Calculated num_node_types: {num_node_types}")
    print(f"Calculated num_edge_types: {num_edge_types}")

    print("\n--- Step 3: Creating fake training artifacts... ---")
    ckpts = {tgt: create_fake_checkpoint(tgt, base_cfg_path, num_node_types, num_edge_types) for tgt in TARGETS}

    print("\n--- Step 4: Running the prediction logic with the fix... ---")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    for idx, tgt in enumerate(TARGETS):
        print(f"\n--- Processing target: {tgt} ---")
        set_cfg(C)
        C.set_new_allowed(True)
        
        cfg_path = ckpts[tgt].parent.parent.parent / f"cfg_{tgt}.yaml"
        print(f"Attempting to load config from: {cfg_path}")
        args_load = argparse.Namespace(cfg_file=str(cfg_path), repeat=1, mark_done=False, opts=[])
        load_cfg(C, args_load)

        model = create_model().to(device)
        
        # This is the line we want to test directly
        print(f"Attempting to load checkpoint with torch.load(...)")
        try:
            model_state = torch.load(ckpts[tgt], map_location=device)['model_state']
            model.load_state_dict(model_state)
            print(f"Successfully loaded checkpoint for target '{tgt}'! The fix is effective.")
        except Exception as e:
            print(f"TEST FAILED: Loading checkpoint failed with error: {e}")
            raise

if __name__ == "__main__":
    main()