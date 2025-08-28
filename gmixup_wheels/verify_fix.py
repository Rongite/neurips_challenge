import json

# Read the notebook
with open('/home/ubuntu/LLM-inference/jikai-project/neurips_challenge/grit-original.ipynb', 'r') as f:
    notebook = json.load(f)

# Get Cell 5 (index 5)
if len(notebook['cells']) > 5:
    cell5 = notebook['cells'][5]
    if 'source' in cell5:
        source_text = ''.join(cell5['source'])
        
        # Check if the problematic G-Mixup arguments are still there
        if '--enable_gmixup' in source_text:
            print("❌ ERROR: G-Mixup arguments still present in Cell 5")
            
            # Find the lines with the problematic code
            lines = source_text.split('\n')
            problem_lines = []
            for i, line in enumerate(lines):
                if any(arg in line for arg in ['--enable_gmixup', '--gmixup_aug_ratio', '--gmixup_lambda_min', '--gmixup_lambda_max']):
                    problem_lines.append(f"Line {i+1}: {line.strip()}")
            
            if problem_lines:
                print("Problematic lines found:")
                for line in problem_lines:
                    print(f"  {line}")
            
        else:
            print("✅ SUCCESS: G-Mixup command-line arguments have been removed from Cell 5")
            
            # Check that the fix comment is present
            if "G-Mixup arguments removed" in source_text and "handled via config file" in source_text:
                print("✅ Explanation comment correctly added")
            else:
                print("⚠️  Explanation comment not found")
            
            # Verify the pipeline_args line is clean
            if "pipeline_args = ['full_pipeline_kaggle.py', '--cfg', str(dynamic_config_path), '--device', device]" in source_text:
                print("✅ Pipeline arguments correctly simplified to basic args only")
            else:
                print("⚠️  Pipeline arguments line not as expected")
                
    else:
        print("Cell 5 has no source content")
else:
    print(f"Notebook has only {len(notebook['cells'])} cells, Cell 5 doesn't exist")