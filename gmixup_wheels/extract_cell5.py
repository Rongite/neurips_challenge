import json

# Read the notebook
with open('/home/ubuntu/LLM-inference/jikai-project/neurips_challenge/grit-original.ipynb', 'r') as f:
    notebook = json.load(f)

# Get Cell 5 (index 5)
if len(notebook['cells']) > 5:
    cell5 = notebook['cells'][5]
    if 'source' in cell5:
        source_text = ''.join(cell5['source'])
        print("=== CELL 5 CONTENT ===")
        print(source_text)
        print("=== END OF CELL 5 ===")
    else:
        print("Cell 5 has no source content")
else:
    print(f"Notebook has only {len(notebook['cells'])} cells, Cell 5 doesn't exist")