
import os
import sys
import subprocess
import glob

def create_dirs(dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created/Verified: {d}")

def generate_ids(data_root, ids_dir):
    # Train IDs
    train_img_dir = os.path.join(data_root, "NYUv2/00_data/train/image")
    if not os.path.exists(train_img_dir):
        # Fallback to local 00_data if absolute path not found or different structure
        # User said: 00_data/NYUv2/... in the recommendation but current list_dir shows 00_data directly in root
        # Let's adjust based on actual structure: /Users/koizumihinata/NYUv2/00_data/train/image
        train_img_dir = "00_data/train/image"
    
    if os.path.isdir(train_img_dir):
        files = sorted(os.listdir(train_img_dir))
        # Filter for images roughly (though spec says sort filenames)
        # Spec says: "Include extension"
        # Ascii ascending
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort() # ASCII ascending
        
        out_path = os.path.join(ids_dir, "train_ids.txt")
        with open(out_path, "w") as f:
            for n in files:
                f.write(n + "\n")
        print(f"Generated {out_path} with {len(files)} IDs")
    else:
        print(f"Warning: {train_img_dir} not found. Skipping train_ids.")

    # Test IDs
    test_img_dir = os.path.join(data_root, "NYUv2/00_data/test/image")
    if not os.path.exists(test_img_dir):
         test_img_dir = "00_data/test/image"

    if os.path.isdir(test_img_dir):
        files = sorted(os.listdir(test_img_dir))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.sort()
        
        out_path = os.path.join(ids_dir, "test_ids.txt")
        with open(out_path, "w") as f:
            for n in files:
                f.write(n + "\n")
        print(f"Generated {out_path} with {len(files)} IDs")
    else:
        print(f"Warning: {test_img_dir} not found. Skipping test_ids.")

def main():
    repo_root = os.getcwd() # Assumes running from /Users/koizumihinata/NYUv2
    
    # 1. Directory Structure
    target_dirs = [
        "00_data/ids",
        "00_data/splits",
        "02_detailcrop",
        "03_nonstruct",
        "04_depth",
        "05_context",
        "03_ensemble"
    ]
    create_dirs(target_dirs)
    
    # 2. Generate IDs
    generate_ids(repo_root, "00_data/ids")
    
    # 3. Export Splits
    # Calling the existing script
    # python -m 01_nearest.research.tools.export_splits --out_dir 00_data/splits
    print("Exporting splits...")
    try:
        cmd = [sys.executable, "01_nearest/research/tools/export_splits.py", "--out_dir", "00_data/splits"]
        # Need to ensure pythonpath includes current dir
        env = os.environ.copy()
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.check_call(cmd, env=env)
        print("Splits exported successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting splits: {e}")

if __name__ == "__main__":
    main()
