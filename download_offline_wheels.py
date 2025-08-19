#!/usr/bin/env python3
"""
Download offline wheels for GRIT environment dependencies in Kaggle:
- Python 3.11.13
- PyTorch 2.6.0+cu124
- CUDA 12.4
- Architecture x86_64
"""

import subprocess
import sys
from pathlib import Path
import shutil

def download_offline_wheels():
    """Download all required wheels for GRIT environment with exact Kaggle specs"""
    
    # Create download directory
    download_dir = Path("/home/ubuntu/LLM-inference/jikai-project/wheels/grit_offline_wheels")
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True)
    
    print("📦 Downloading offline wheels for GRIT environment in Kaggle:")
    print("   Python: 3.11.13")
    print("   PyTorch: 2.6.0+cu124") 
    print("   CUDA: 12.4")
    print("   Arch: x86_64")
    print(f"📂 Download directory: {download_dir}")
    
    # All required packages for GRIT environment
    packages = [
        # PyTorch Geometric and related
        "torch_geometric",
        "torch-scatter", 
        "torch-sparse",
        "torch-cluster", 
        "torch-spline-conv",
        # Core dependencies
        "rdkit",
        "ogb",
        "datasketch",
        "networkx",
        "tqdm",
        "tabulate",
        "opt-einsum",
        "yacs",
        # Scientific computing (CRITICAL - found missing in GRIT code analysis)
        "scipy",           # Used in grit/logger.py and grit/transform/hashes/hash_dataset.py
        "scikit-learn",    # Used in grit/logger.py for ML metrics
        "numpy",           # Used extensively throughout GRIT
        "matplotlib",      # Used in full_pipeline.py for plotting
        "torchmetrics",    # Used in grit/logger.py for AUROC
        # Additional dependencies
        "requests",
        "fsspec",
        "cloudpickle",
        "jinja2",
        "pyparsing",
        "typing-extensions",
        # Network/async
        "aiosignal",
        "async-timeout",
        "aiohappyeyeballs",
        "attrs",
        "certifi",
        "idna",
        "urllib3"
    ]
    
    successful_downloads = []
    failed_downloads = []
    
    for package in packages:
        print(f"\n⬇️ Downloading {package} for Kaggle x86_64...")
        
        # Strategy 1: Specialized handling for different package types
        try:
            if package.startswith("torch") and package != "torch_geometric":
                # For torch-scatter, torch-sparse, etc. - use PyG index with cu124
                cmd = [
                    sys.executable, "-m", "pip", "download",
                    package,
                    "--dest", str(download_dir),
                    "--only-binary=:all:",
                    "--platform", "linux_x86_64", 
                    "--python-version", "311",  # Use 311 instead of 3.11 for PyG
                    "--index-url", "https://data.pyg.org/whl/torch-2.6.0+cu124.html",
                    "--trusted-host", "data.pyg.org",
                    "--no-deps"
                ]
                print(f"  🔄 Strategy 1: PyG cu124 index (x86_64, py311)")
            elif package == "torch_geometric":
                # torch_geometric from regular PyPI
                cmd = [
                    sys.executable, "-m", "pip", "download",
                    package,
                    "--dest", str(download_dir),
                    "--only-binary=:all:",
                    "--platform", "linux_x86_64",
                    "--python-version", "3.11",
                    "--no-deps"
                ]
                print(f"  🔄 Strategy 1: PyPI (x86_64, py3.11)")
            elif package == "rdkit":
                # rdkit - force x86_64 platform
                cmd = [
                    sys.executable, "-m", "pip", "download",
                    "rdkit-pypi",  # Use rdkit-pypi which has better wheel support
                    "--dest", str(download_dir),
                    "--only-binary=:all:",
                    "--platform", "linux_x86_64",
                    "--python-version", "3.11",
                    "--no-deps"
                ]
                print(f"  🔄 Strategy 1: PyPI rdkit-pypi (x86_64, py3.11)")
            else:
                # Regular packages from PyPI
                cmd = [
                    sys.executable, "-m", "pip", "download",
                    package,
                    "--dest", str(download_dir),
                    "--only-binary=:all:",
                    "--platform", "linux_x86_64",
                    "--python-version", "3.11",
                    "--no-deps"
                ]
                print(f"  🔄 Strategy 1: PyPI (x86_64, py3.11)")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✅ {package} downloaded successfully!")
                successful_downloads.append(package)
                continue
            else:
                print(f"  ❌ Strategy 1 failed: {result.stderr.strip()[:100]}")
                
        except Exception as e:
            print(f"  ❌ Strategy 1 error: {e}")
        
        # Strategy 2: PyG index without platform constraints
        try:
            cmd = [
                sys.executable, "-m", "pip", "download",
                package,
                "--dest", str(download_dir),
                "--extra-index-url", "https://data.pyg.org/whl/torch-2.6.0+cu124.html",
                "--trusted-host", "data.pyg.org",
                "--no-deps"
            ]
            
            print(f"  🔄 Strategy 2: PyG index (no platform constraints)")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✅ {package} downloaded!")
                successful_downloads.append(package)
                continue
            else:
                print(f"  ❌ Strategy 2 failed: {result.stderr.strip()[:100]}")
                
        except Exception as e:
            print(f"  ❌ Strategy 2 error: {e}")
        
        # Strategy 3: Regular PyPI with platform specs
        try:
            cmd = [
                sys.executable, "-m", "pip", "download",
                package,
                "--dest", str(download_dir),
                "--only-binary=:all:",
                "--platform", "linux_x86_64",
                "--python-version", "3.11",
                "--no-deps"
            ]
            
            print(f"  🔄 Strategy 3: PyPI (x86_64, py3.11)")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✅ {package} downloaded from PyPI!")
                successful_downloads.append(package)
                continue
            else:
                print(f"  ❌ Strategy 3 failed: {result.stderr.strip()[:100]}")
                
        except Exception as e:
            print(f"  ❌ Strategy 3 error: {e}")
        
        # If all strategies failed
        print(f"  ❌ All strategies failed for {package}")
        failed_downloads.append(package)
    
    # Analyze downloaded files
    print("\n📋 Downloaded files:")
    wheel_files = list(download_dir.glob("*.whl"))
    tar_files = list(download_dir.glob("*.tar.gz"))
    total_size = 0
    
    for file in wheel_files + tar_files:
        size_mb = file.stat().st_size / (1024 * 1024) 
        total_size += size_mb
        
        # Check if it's x86_64 compatible
        if file.suffix == '.whl':
            if 'x86_64' in file.name or 'linux_x86_64' in file.name:
                arch_status = "✅ x86_64"
            elif 'any' in file.name:
                arch_status = "✅ universal"
            elif 'aarch64' in file.name or 'arm64' in file.name:
                arch_status = "❌ ARM64"
            else:
                arch_status = "❓ unknown"
        else:
            arch_status = "📦 source"
            
        print(f"  📄 {file.name} ({size_mb:.1f} MB) - {arch_status}")
    
    print(f"\n📊 Download Summary:")
    print(f"✅ Successful: {len(successful_downloads)} packages")
    print(f"❌ Failed: {len(failed_downloads)} packages") 
    print(f"📦 Total files: {len(wheel_files + tar_files)}")
    print(f"💾 Total size: {total_size:.1f} MB")
    
    if successful_downloads:
        print(f"\n✅ Successfully downloaded:")
        for pkg in successful_downloads:
            print(f"  - {pkg}")
    
    if failed_downloads:
        print(f"\n❌ Failed to download:")
        for pkg in failed_downloads:
            print(f"  - {pkg}")
    
    print(f"\n🎯 Next steps:")
    print(f"1. Copy wheels to your offline dataset directory")
    print(f"2. Update dataset zip file") 
    print(f"3. Upload to Kaggle")
    print(f"4. Install with: pip install --find-links /kaggle/input/dataset *.whl")
    
    return download_dir, successful_downloads, failed_downloads

if __name__ == "__main__":
    download_offline_wheels()