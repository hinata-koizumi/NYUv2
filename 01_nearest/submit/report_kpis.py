
import os
import argparse
import numpy as np
import json
import cv2
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Constants
STRUCT_IDS = [0, 3, 4, 8, 9, 11, 12]
NONSTRUCT_IDS = [1, 2, 5, 6, 7, 10]
CID_TABLE = 9
CID_WALL = 11
CID_FURNITURE = 5

def process_single_sample(args):
    fid, l, data_root, kernel = args
    
    # Load GT
    if fid.endswith(".png"):
        gt_path = os.path.join(data_root, fid)
    else:
        gt_path = os.path.join(data_root, f"{fid}.png")
        
    if not os.path.exists(gt_path):
        # Fallback check
        pass
             
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt is None: return None
    if gt.ndim == 3: gt = gt[:,:,0]
    
    # Resize Logic if needed
    if l.shape[1:] != gt.shape:
        # We need to resize l (logits)
        # Using pytorch for clean bilinear
        l_t = torch.from_numpy(l).unsqueeze(0).float() # (1, C, H, W)
        l_t = torch.nn.functional.interpolate(l_t, size=gt.shape, mode='bilinear', align_corners=False)
        l = l_t.squeeze(0).numpy()
        
    pred = np.argmax(l, axis=0).astype(np.uint8)
    valid = (gt != 255)
    
    res = {}
    
    # CM
    cm = np.zeros((13, 13), dtype=np.int64)
    if valid.any():
        idx = pred[valid].astype(np.int64) * 13 + gt[valid].astype(np.int64)
        c = np.bincount(idx, minlength=169)
        cm += c.reshape(13, 13)
    res["cm"] = cm
    
    # Boundary
    v_u8 = valid.astype(np.uint8)
    safe = cv2.erode(v_u8, kernel)
    g_u8 = gt.astype(np.uint8)
    d = cv2.dilate(g_u8, kernel)
    e = cv2.erode(g_u8, kernel)
    grad = (d != e)
    b_gt = (grad & (safe > 0)) # boundary map
    
    d_p = cv2.dilate(pred, kernel)
    e_p = cv2.erode(pred, kernel)
    b_p = (d_p != e_p) & valid
    
    res["b_inter"] = np.sum(b_gt & b_p)
    res["b_union"] = np.sum(b_gt | b_p)
    
    # Struct Precision/Recall
    p_struct = np.isin(pred, STRUCT_IDS)
    g_struct = np.isin(gt, STRUCT_IDS)
    
    mask = valid
    
    res["tp_struct"] = np.sum(p_struct & g_struct & mask)
    res["fp_struct"] = np.sum(p_struct & (~g_struct) & mask)
    res["fn_struct"] = np.sum((~p_struct) & g_struct & mask)
    
    return res

def report_kpis(exp_dir, protocol="E1", dump_confusion=False):
    # Paths
    oof_dir = os.path.join(exp_dir, "01_nearest/golden_artifacts/oof")
    if not os.path.exists(oof_dir):
        if os.path.exists(os.path.join(exp_dir, "golden_artifacts/oof")):
            oof_dir = os.path.join(exp_dir, "golden_artifacts/oof")
            
    logits_path = os.path.join(oof_dir, "oof_logits.npy")
    ids_path = os.path.join(oof_dir, "oof_file_ids.npy")
    
    if not os.path.exists(logits_path):
        print(f"Error: {logits_path} not found.")
        return

    print(f"Loading OOF from {logits_path} (mmap)...")
    logits_all = np.load(logits_path, mmap_mode='r')
    ids_all = np.load(ids_path)
    
    data_root = "/root/datasets/NYUv2/00_data/train/label"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Prepare Tasks
    # IMPORTANT: Accessing mmap in threads can be tricky if not careful, 
    # but reading different slices is usually fine.
    # We pass the SLICE (copy) to the thread? No, passing the mmap object and index is risky for pickling?
    # Actually ThreadPool shares memory.
    # But to be safe and avoid lock contention on single mmap handle, we might read data in main thread?
    # No, that blocks.
    # Let's try passing the index and let the function read from global or passed array.
    
    tasks = []
    # Hack: We can't pickle mmap object easily for ProcessPool, but ThreadPool is fine.
    # However, to be extra safe with mmap concurrently, we'll see.
    # If it fails, we chunk.
    
    # But accessing numpy mmap from multiple threads usually works fine for reading.
    for i in range(len(ids_all)):
        # Construct args. Passing logits_all[i] might force a read? 
        # No, it returns a view or triggers read. 
        # Ideally we want the read to happen in the thread.
        # But slicing mmap returns a new array/view.
        tasks.append((ids_all[i], logits_all[i], data_root, kernel))
        
    print(f"Processing {len(tasks)} samples with ThreadPool (8 workers)...")
    
    # Accumulators
    total_cm = np.zeros((13, 13), dtype=np.int64)
    total_bi = 0
    total_bu = 0
    total_tps = 0
    total_fps = 0
    total_fns = 0
    
    with ThreadPoolExecutor(max_workers=8) as exe:
        for res in tqdm(exe.map(process_single_sample, tasks), total=len(tasks)):
            if res is None: continue
            total_cm += res["cm"]
            total_bi += res["b_inter"]
            total_bu += res["b_union"]
            total_tps += res["tp_struct"]
            total_fps += res["fp_struct"]
            total_fns += res["fn_struct"]
            
    # Metrics Calc
    tp = np.diag(total_cm)
    fp = np.sum(total_cm, axis=0) - tp
    fn = np.sum(total_cm, axis=1) - tp
    
    iou = tp / (tp + fp + fn + 1e-6)
    
    miou_all = np.mean(iou)
    miou_struct = np.mean(iou[STRUCT_IDS])
    miou_table = iou[CID_TABLE]
    miou_bound = total_bi / (total_bu + 1e-6)
    
    struct_prec = total_tps / (total_tps + total_fps + 1e-6)
    struct_recall = total_tps / (total_tps + total_fns + 1e-6)
    
    metrics = {
        "mIoU_all": float(miou_all),
        "mIoU_struct": float(miou_struct),
        "mIoU_table": float(miou_table),
        "mIoU_boundary": float(miou_bound),
        "struct_precision": float(struct_prec),
        "struct_recall": float(struct_recall)
    }
    
    # Save
    out_dir = os.path.join(exp_dir, "01_nearest/golden_artifacts")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Baseline
    base = {
        "mIoU_all": 0.6638,
        "mIoU_struct": 0.7229,
        "mIoU_table": 0.5070,
        "mIoU_boundary": 0.0380,
        "struct_precision": 0.9223,
        "struct_recall": 0.90
    }
    
    # MD Report
    md = f"# Gate Check Report (Protocol: {protocol})\n\n"
    md += f"| Metric | Baseline | **OOF** | Diff | Gate |\n"
    md += f"| :--- | :--- | :--- | :--- | :--- |\n"
    
    def row(key, b_val, gate_diff=None, high_is_good=True):
        val = metrics[key]
        diff = val - b_val
        diff_str = f"{diff:+.4f}"
        status = "✅"
        if gate_diff is not None:
             if high_is_good:
                 if diff < gate_diff: status = "❌"
             else:
                 if diff > gate_diff: status = "❌"
        return f"| **{key}** | {b_val:.4f} | **{val:.4f}** | {diff_str} | {status} |\n"
        
    md += row("mIoU_all", base["mIoU_all"], -0.002)
    md += row("mIoU_struct", base["mIoU_struct"], 0.01)
    md += row("mIoU_table", base["mIoU_table"], 0.02)
    md += row("mIoU_boundary", base["mIoU_boundary"], 0.003)
    md += row("struct_precision", base["struct_precision"], -0.003)
    
    status_rec = "✅" if metrics["struct_recall"] >= 0.90 else "❌"
    md += f"| **struct_recall** | {(base['struct_recall']):.2f} (Goal) | **{metrics['struct_recall']:.4f}** | - | {status_rec} |\n"
    
    with open(os.path.join(out_dir, "metrics.md"), "w") as f:
        f.write(md)
        
    print(md)
    
    if dump_confusion:
        print("\n=== Confusion Audit ===")
        total_table = np.sum(total_cm[CID_TABLE, :])
        tbl_to_furn = total_cm[CID_TABLE, CID_FURNITURE]
        rate_tf = tbl_to_furn / (total_table + 1e-6)
        print(f"Table -> Furniture: {rate_tf*100:.2f}%")
        
        total_wall = np.sum(total_cm[CID_WALL, :])
        wall_to_furn = total_cm[CID_WALL, CID_FURNITURE]
        rate_wf = wall_to_furn / (total_wall + 1e-6)
        print(f"Wall -> Furniture : {rate_wf*100:.2f}%")
        print("=======================")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", type=str, required=True)
    p.add_argument("--protocol", type=str, default="E1")
    p.add_argument("--dump-confusion", action="store_true")
    args = p.parse_args()
    report_kpis(args.exp_dir, args.protocol, args.dump_confusion)
