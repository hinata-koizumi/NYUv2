import os
import torch
import numpy as np
import zipfile
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.base_config import Config
from src.data.dataset import NYUDataset
from src.data.transforms import get_valid_transforms
from src.data.adapters import get_adapter
from src.model.meta_arch import SegFPN
from src.engine.inference import Predictor
from src.utils.misc import seed_everything, worker_init_fn

def collect_fold_weights(output_dir: str, n_folds: int):
    paths = []
    for f in range(n_folds):
        p = os.path.join(output_dir, f"fold{f}", "model_best.pth")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing weights: {p}")
        paths.append(p)
    return paths

def main():
    cfg = Config()
    seed_everything(cfg.SEED)

    output_dir = os.path.join("data", "outputs", cfg.EXP_NAME)
    weight_paths = collect_fold_weights(output_dir, cfg.N_FOLDS)
    
    test_image_dir = os.path.join(cfg.TEST_DIR, "image")
    test_depth_dir = os.path.join(cfg.TEST_DIR, "depth")
    
    # Get Test Files
    image_files = sorted([os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith(".png")])
    if len(image_files) == 0:
        raise FileNotFoundError("No test images found.")
        
    depth_files = []
    for img_p in image_files:
        base = os.path.basename(img_p)
        d_p = os.path.join(test_depth_dir, base)
        depth_files.append(d_p if os.path.exists(d_p) else None)
        
    # Dataset
    adapter = get_adapter(cfg)
    test_ds = NYUDataset(
        image_paths=np.array(image_files),
        label_paths=None,
        depth_paths=np.array(depth_files),
        cfg=cfg,
        transform=get_valid_transforms(cfg),
        color_transform=None,
        enable_smart_crop=False,
        adapter=adapter,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE * 2, # Faster inference
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    
    # Check output size using one sample logic? 
    # Or just assume 480x640 for NYUv2.
    # But code should be generic.
    # We will accumulate into a LIST of memmaps per file? 
    # Or just one giant memmap? 
    # Since sizes are all same for NYUv2 (480x640), giant memmap is fine.
    # N x H x W x C
    N = len(image_files)
    H, W = 480, 640 # NYUv2 Standard. 
    # Wait, if `inference.py` returns `orig_h`, we trust it.
    # But to allocate MemMap, we need shape.
    # We can peek at first sample meta.
    _, _, _, _, meta0 = test_ds[0]
    H_orig, W_orig = meta0['orig_h'], meta0['orig_w']
    
    os.makedirs("tmp", exist_ok=True)
    mm_path = f"tmp/probs_accum_{int(time.time())}.dat"
    # Accumulate PROBS or LOGITS?
    # LOGITS!
    # User Recommendation: Accum Logits in Float32.
    acc = np.memmap(mm_path, dtype="float32", mode="w+", shape=(N, H_orig, W_orig, cfg.NUM_CLASSES))
    acc[:] = 0.0
    acc.flush()
    
    # Iterate Folds
    for wp in weight_paths:
        print(f"Processing {wp}...")
        model = SegFPN(num_classes=cfg.NUM_CLASSES, in_channels=cfg.IN_CHANNELS)
        state = torch.load(wp, map_location="cpu")
        model.load_state_dict(state)
        model.to(cfg.DEVICE)
        model.eval()
        
        predictor = Predictor(model, test_loader, cfg.DEVICE, cfg)
        
        # Predict Logits (Generator)
        idx = 0
        # Use simple mean temperature or ensemble-aware?
        # User said: "predict_with_tta ... iterates TTA ... applies Temp"
        # We can use best temp from config.
        # Temp should be applied *inside* Predictor per TTA step.
        # But for Ensembling Folds:
        # Sum( Logits(M1) + Logits(M2) ... ) / N
        # Logits(M1) = Mean( TTA_Logits / Temp )
        # So yes, Predictor handles Temp and TTA averaging.
        # We just sum Predictor outputs.
        
        temp = cfg.TEMPERATURES[0] # Simplification: Use first or fixed. 
        # Refinement: In 093.5 we selected best temp. 
        # Here we assume fixed for simplicity or user can tune.
        
        for logits_item in predictor.predict_logits(temperature=temp):
            # logits_item: (H, W, C)
            acc[idx] += logits_item
            idx += 1
            
        acc.flush()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # Normalize by N_FOLDS
    # Actually for Argmax, division doesn't change relative order.
    # So we can skip division.
    
    print("Generating submission...")
    preds = []
    for i in tqdm(range(N)):
        p = acc[i] # (H, W, C) Logits
        # Argmax
        cls = np.argmax(p, axis=2).astype(np.uint8)
        preds.append(cls)
        
    preds = np.array(preds)
    # Save .npy
    np.save("tmp/submission.npy", preds)
    
    # Zip
    with zipfile.ZipFile("submission.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write("tmp/submission.npy", arcname="tmp/submission.npy")
        
    print("Done! submission.zip created.")
    
    # Cleanup
    if os.path.exists(mm_path):
        os.remove(mm_path)

if __name__ == "__main__":
    main()
