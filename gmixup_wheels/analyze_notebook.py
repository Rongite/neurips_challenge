import json
import sys

# Read the notebook
notebook_path = '/home/ubuntu/LLM-inference/jikai-project/neurips_challenge/grit-original.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# Search through cells for gmixup or pipeline related content
found_cells = []
for i, cell in enumerate(notebook.get('cells', [])):
    if 'source' in cell:
        source_text = ''.join(cell['source'])
        # Look for various patterns
        if any(keyword in source_text for keyword in [
            'gmixup', 'full_pipeline', 'enable_gmixup', 
            'subprocess', 'os.system', '!python',
            '--enable_gmixup', '--gmixup_aug_ratio',
            '--gmixup_lambda_min', '--gmixup_lambda_max'
        ]):
            found_cells.append((i, source_text))

# Print found cells
print(f"Found {len(found_cells)} relevant cells:")
for i, source in found_cells:
    print(f'\n=== CELL INDEX {i} ===')
    print(source)
    print('=' * 50)

# Also search for any cell that contains command line arguments pattern
all_cells_with_args = []
for i, cell in enumerate(notebook.get('cells', [])):
    if 'source' in cell:
        source_text = ''.join(cell['source'])
        if '--' in source_text and any(word in source_text for word in ['python', 'run', 'exec']):
            all_cells_with_args.append((i, source_text))

if all_cells_with_args:
    print(f"\n\nAdditionally found {len(all_cells_with_args)} cells with command-line arguments:")
    for i, source in all_cells_with_args:
        print(f'\n=== CELL INDEX {i} (with args) ===')
        print(source)
        print('=' * 50)