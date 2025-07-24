"""
This script orchestrates the end-to-end workflow for training GRIT models
to predict polymer properties for the NeurIPS Open Polymer Prediction 2025 competition.

It performs the following steps:
1.  Loads and consolidates the main training data with all supplemental files.
2.  Canonicalizes all SMILES strings to ensure a standard representation.
3.  Calls the graph conversion logic to transform SMILES into PyTorch Geometric graphs.
4.  For each of the 5 target properties, launches the GRIT training script (`GRIT/main.py`)
    with a suitable configuration, creating one model per target.
5.  Saves the trained models and results in a structured output directory.

Dependencies:
- pandas
- rdkit-pypi
- torch
- torch_geometric
"""
import pandas as pd
import numpy as np
from rdkit import Chem
import os
import subprocess

# --- Import graph conversion functions from the user-provided script ---
# This assumes `convert_smiles_to_graphs.py` is in the same directory.
from convert_smiles_to_graphs import smiles_to_graph_list, save_graph_list

# --- Constants and Configuration ---
# File Paths
BASE_DATA_PATH = '/home/ubuntu/LLM-inference/jikai-project/neurips_challenge/data'
TRAIN_CSV_PATH = os.path.join(BASE_DATA_PATH, 'train.csv')
SUPPLEMENT_PATH = os.path.join(BASE_DATA_PATH, 'train_supplement')
GRIT_TRAIN_SCRIPT = '/home/ubuntu/LLM-inference/jikai-project/neurips_challenge/GRIT/main.py'

# Output Directories
OUTPUT_DIR = 'grit_training_output'
PROCESSED_DATA_DIR = os.path.join(OUTPUT_DIR, 'processed_data')
GRAPH_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_graphs.pt')
CONSOLIDATED_CSV_PATH = os.path.join(PROCESSED_DATA_DIR, 'consolidated_training_data.csv')

# Target properties to predict
TARGET_COLUMNS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# --- Helper Functions ---

def canonicalize_smiles(s):
    """
    Converts a SMILES string to its canonical form using RDKit.
    Returns None if the SMILES string is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)
        else:
            return None
    except:
        return None

def load_and_consolidate_data():
    """
    Loads the main train.csv and all supplemental data, canonicalizes SMILES,
    and consolidates them into a single DataFrame.
    """
    print("1. Loading and consolidating data...")
    
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    train_df['SMILES'] = train_df['SMILES'].apply(canonicalize_smiles)
    
    supp1_df = pd.read_csv(os.path.join(SUPPLEMENT_PATH, 'dataset1.csv')).rename(columns={'TC_mean': 'Tc'})
    supp2_df = pd.read_csv(os.path.join(SUPPLEMENT_PATH, 'dataset2.csv'))
    supp3_df = pd.read_csv(os.path.join(SUPPLEMENT_PATH, 'dataset3.csv'))
    supp4_df = pd.read_csv(os.path.join(SUPPLEMENT_PATH, 'dataset4.csv'))

    supplemental_dfs = [supp1_df, supp2_df, supp3_df, supp4_df]
    
    for i, df in enumerate(supplemental_dfs):
        if 'SMILES' in df.columns:
            df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)

    combined_df = pd.concat([train_df] + supplemental_dfs, ignore_index=True, sort=False)
    combined_df.dropna(subset=['SMILES'], inplace=True)
    
    numeric_cols = combined_df.select_dtypes(include=np.number).columns.tolist()
    group_cols = ['SMILES'] + numeric_cols
    consolidated_df = combined_df[group_cols].groupby('SMILES').mean().reset_index()
    
    print(f"Data consolidated. Total unique polymers: {len(consolidated_df)}")
    consolidated_df.to_csv(CONSOLIDATED_CSV_PATH, index=False)
    print(f"Consolidated data saved to '{CONSOLIDATED_CSV_PATH}'")
    
    return consolidated_df

def prepare_graph_data(smiles_series, labels_df):
    """
    Converts SMILES to graphs and saves them along with their labels.
    """
    print("\n2. Converting SMILES to graph data...")
    graph_list = smiles_to_graph_list(smiles_series)
    
    # The graph list must be saved with labels for the GRIT framework
    # We will attach the entire labels dataframe to the saved object.
    save_graph_list(graph_list, GRAPH_DATA_PATH, y_df=labels_df)
    print(f"Graph data saved to '{GRAPH_DATA_PATH}'")

def main():
    """
    Main function to run the full data prep and GRIT training pipeline.
    """
    # Ensure all output directories exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Step 1: Create the master dataset with all labels
    master_df = load_and_consolidate_data()
    
    # Step 2: Convert all SMILES in the master dataset to graphs
    # The labels (y) are passed along and saved with the graphs
    prepare_graph_data(master_df['SMILES'], master_df[TARGET_COLUMNS])

    print("\n3. Starting GRIT model training for each target...")
    # Step 3: Loop through each target and launch the GRIT training script
    for target in TARGET_COLUMNS:
        print("-" * 60)
        print(f"LAUNCHING GRIT TRAINING FOR TARGET: {target}")
        print("-" * 60)

        # Define a unique output directory for this model's results
        model_out_dir = os.path.join(OUTPUT_DIR, f'grit_model_{target}')
        os.makedirs(model_out_dir, exist_ok=True)

        # Command to execute the GRIT training script
        # This uses a configuration that is well-suited for molecular regression.
        # It tells the GRIT script:
        #   --cfg: Which config file to use
        #   dataset.name: A custom name for our dataset
        #   dataset.format: To use our PyG (PyTorch Geometric) file
        #   dataset.dir: Where to find the graph file
        #   dataset.task_type: To perform regression
        #   train.y_col_name: Which column in our saved dataframe to use as the label
        #   out_dir: Where to save the results (checkpoints, logs, etc.)
        
        cmd = [
            'python',
            GRIT_TRAIN_SCRIPT,
            '--cfg', 'configs/GRIT/peptides-func-GRIT-RRWP.yaml',
            'dataset.name', 'Polymers',
            'dataset.format', 'PyG-custom',
            'dataset.dir', f'{PROCESSED_DATA_DIR}',
            'dataset.task_type', 'regression',
            'train.y_col_name', target,
            'out_dir', model_out_dir
        ]

        print(f"Executing command:\n{' '.join(cmd)}")

        # Launch the subprocess
        # We run this from the GRIT directory to ensure it finds its internal modules
        process = subprocess.Popen(cmd, cwd='GRIT', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Stream the output from the training script to the console
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        rc = process.poll()
        if rc == 0:
            print(f"\nSuccessfully completed training for {target}.")
        else:
            print(f"\nError during training for {target}. Return code: {rc}")

    print("-" * 60)
    print("\nFull training pipeline complete.")

if __name__ == '__main__':
    main()