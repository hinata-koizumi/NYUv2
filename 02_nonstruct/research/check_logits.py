import numpy as np
import os

MODEL_A_LOGITS = "/root/datasets/NYUv2/00_data/output/01_nearest_v1.0_frozen/golden_artifacts/oof_logits.npy"
MODEL_B_LOGITS = "/root/datasets/NYUv2/02_nonstruct/output/oof_fold0_logits.npy"

def check_stats(path, name):
    print(f"\nChecking {name} ({path}):")
    data = np.load(path, mmap_mode='r')
    sample = data[0]
    print(f"  Shape: {data.shape}")
    print(f"  Min:   {np.min(sample):.4f}")
    print(f"  Max:   {np.max(sample):.4f}")
    print(f"  Mean:  {np.mean(sample):.4f}")
    print(f"  Std:   {np.std(sample):.4f}")
    
    # Check if softmaxed?
    exp_sum = np.sum(np.exp(sample - np.max(sample, axis=0)), axis=0)
    # This is not a good check if it was softmaxed before saving.
    # Just check if across channels it sums to ~1.
    sum_across = np.sum(sample, axis=0)
    print(f"  Sum across channels (mean): {np.mean(sum_across):.4f}")
    print(f"  Sum across channels (std):  {np.std(sum_across):.4f}")

check_stats(MODEL_A_LOGITS, "Model A")
check_stats(MODEL_B_LOGITS, "Model B")
