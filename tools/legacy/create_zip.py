
import os
import zipfile
import shutil
import glob

def create_submission_zip():
    base_dir = "/root/datasets/NYUv2"
    submission_npy = os.path.join(base_dir, "submission.npy")
    zip_path = os.path.join(base_dir, "submission.zip")
    
    if not os.path.exists(submission_npy):
        print(f"Error: {submission_npy} not found!")
        # Try checking data/output just in case, though ls output will confirm
        return

    weight_files = []
    for i in range(5):
        w_path = os.path.join(base_dir, "data/output/nearest_final", f"fold{i}", "model_best.pth")
        if not os.path.exists(w_path):
            print(f"Error: {w_path} not found!")
            return
        weight_files.append((w_path, f"weights/fold{i}.pt"))

    print(f"Creating {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add submission.npy
        print(f"  Adding submission.npy")
        zf.write(submission_npy, "submission.npy")
        
        # Add weights
        for src, dst in weight_files:
            print(f"  Adding {src} -> {dst}")
            zf.write(src, dst)
            
    print("Optimization: Done.")
    
    # Verify
    print("\nVerifying zip content:")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    create_submission_zip()
