#!/usr/bin/env python3
"""
Create neurips-offline-wheels-truly-offline.zip package
Step 3 of the offline wheels pipeline: pack all wheels with install_offline.py
"""

import shutil
import subprocess
import sys
from pathlib import Path
import zipfile

def create_offline_zip():
    """Create the complete offline wheels ZIP package"""
    
    print("🚀 Creating neurips-offline-wheels-truly-offline.zip")
    print("=" * 60)
    
    # Paths
    script_dir = Path(__file__).resolve().parent
    wheels_dir = script_dir.parent / "wheels" / "grit_offline_wheels"
    output_zip = script_dir / "neurips-offline-wheels-truly-offline.zip"
    install_script = script_dir / "install_offline.py"
    
    print(f"📂 Wheels directory: {wheels_dir}")
    print(f"📜 Install script: {install_script}")
    print(f"📦 Output ZIP: {output_zip}")
    
    # Step 1: Verify install script exists
    if not install_script.exists():
        print(f"❌ Install script not found: {install_script}")
        return False
    
    # Step 2: Check if wheels directory exists
    if not wheels_dir.exists():
        print(f"\n⚠️ Wheels directory not found: {wheels_dir}")
        print("💡 Run download_offline_wheels.py first to download wheels")
        return False
    
    # Step 3: Count available wheel files
    wheel_files = list(wheels_dir.glob("*.whl"))
    print(f"\n📊 Found {len(wheel_files)} wheel files")
    
    if len(wheel_files) == 0:
        print("❌ No wheel files found!")
        print("💡 Run download_offline_wheels.py first to download wheels")
        return False
    
    # Step 4: Remove existing ZIP if it exists
    if output_zip.exists():
        output_zip.unlink()
        print(f"🗑️ Removed existing ZIP: {output_zip}")
    
    # Step 5: Create new ZIP file
    print(f"\n📦 Creating new ZIP file...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add install script first
        zipf.write(install_script, "install_offline.py")
        print(f"  ✅ Added: install_offline.py")
        
        # Add all wheel files
        for wheel_file in sorted(wheel_files):
            zipf.write(wheel_file, wheel_file.name)
            print(f"  ✅ Added: {wheel_file.name}")
    
    # Step 6: Verify and report ZIP file details
    zip_size_mb = output_zip.stat().st_size / (1024 * 1024)
    
    print(f"\n📊 ZIP File Created Successfully!")
    print(f"📁 Location: {output_zip}")
    print(f"💾 Size: {zip_size_mb:.1f} MB")
    print(f"📦 Total files: {len(wheel_files) + 1}")
    print(f"   - 1 installation script (install_offline.py)")
    print(f"   - {len(wheel_files)} wheel files")
    
    # Step 7: Verify ZIP contents
    print(f"\n🔍 Verifying ZIP contents...")
    with zipfile.ZipFile(output_zip, 'r') as zipf:
        zip_contents = zipf.namelist()
        
        # Check install script
        if "install_offline.py" in zip_contents:
            print("  ✅ install_offline.py present")
        else:
            print("  ❌ install_offline.py missing!")
            return False
        
        # Check wheels
        wheel_count = sum(1 for name in zip_contents if name.endswith('.whl'))
        print(f"  ✅ {wheel_count} wheel files verified")
        
        # Show key packages
        key_packages = []
        for name in zip_contents:
            if name.endswith('.whl'):
                package = name.split('-')[0]
                if package in ['torch_geometric', 'networkx', 'tqdm', 'yacs', 'requests']:
                    key_packages.append(name)
        
        if key_packages:
            print(f"  📋 Key packages found:")
            for pkg in sorted(key_packages):
                print(f"    - {pkg}")
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Upload {output_zip.name} to Kaggle competition notebook")
    print(f"2. Add as dataset and name it: 'grit-wheels'")
    print(f"3. In notebook, run: exec(open('/kaggle/input/grit-wheels/install_offline.py').read())")
    print(f"4. All dependencies will be installed offline!")
    
    print(f"\n🎉 SUCCESS! neurips-offline-wheels-truly-offline.zip is ready for upload")
    
    return True

def main():
    """Main function with error handling"""
    try:
        success = create_offline_zip()
        if success:
            print(f"\n✅ ZIP creation completed successfully")
            return 0
        else:
            print(f"\n❌ ZIP creation failed")
            return 1
            
    except Exception as e:
        print(f"\n💥 Error during ZIP creation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())