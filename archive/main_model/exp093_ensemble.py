import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
from tqdm import tqdm
import cv2
from scipy.optimize import minimize

# --- Configuration ---
class Config:
    EXP_NAME = "exp093_ensemble_4models"
    SEED = 42
    
    # Image Settings
    # Base resolution for Inference (matches training Crop Size)
    # Training: Resize 720x960 -> Crop 576x768
    # Validation in base script: Resize 576x768
    # So we should resize input to 576x768 before TTA
    BASE_HEIGHT = 576
    BASE_WIDTH = 768
    
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    
    # Fold
    N_FOLDS = 5
    
    # Depth range for 093.5
    DEPTH_MIN = 0.71  
    DEPTH_MAX = 10.0  
    
    # Normalization constants (RGB)
    MEAN = [0.485, 0.456, 0.406] 
    STD = [0.229, 0.224, 0.225]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    DATA_ROOT = 'data/train'
    
    # Models to Ensemble
    MODELS = [
        # 093.2: RGB-FPN (ファイルが破損しているためコメントアウト)
        # {
        #     "name": "exp093_2",
        #     "path_template": "data/outputs/exp093_2_hr_ema_tta_oof/fold{}/model_best.pth",
        #     "type": "FPN",
        #     "in_channels": 3
        # },
        # 093.5: RGB-D FPN
        {
            "name": "exp093_5",
            "path_template": "data/outputs/exp093_5_convnext_rgbd_4ch/fold{}/model_best.pth",
            "type": "FPN",
            "in_channels": 4
        },
        # 093.4: Boundary FPN
        {
            "name": "exp093_4",
            "path_template": "data/outputs/exp093_4_boundary_cb/fold{}/model_best.pth",
            "type": "FPN",
            "in_channels": 3
        },
        # 093.6: DeepLabV3+ (モデルファイルが存在しないためコメントアウト)
        # {
        #     "name": "exp093_6",
        #     "path_template": "data/outputs/exp093_6_deeplabv3p_convnext/fold{}/model_best.pth",
        #     "type": "DeepLabV3Plus",
        #     "in_channels": 3
        # }
    ]
    
    # TTA Settings (Reduced)
    TTA_COMBS = [
        (0.75, False), (0.75, True),
        (1.0, False),  (1.0, True)
    ] 
    
    INFERENCE_TEMP = 0.7
    
    # Memory Optimization: Resize factors for storing probabilities
    # Relative to the INFERENCE resolution (BASE_HEIGHT/WIDTH)
    STORE_SCALE = 0.5


# --- Model Definitions ---
# 1. FPN
class MultiTaskFPN(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.backbone = smp.FPN(
            encoder_name="tu-convnext_base", 
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
        try:
            decoder_channels = self.backbone.decoder.out_channels
        except AttributeError:
            decoder_channels = self.backbone.segmentation_head[0].in_channels

        self.depth_head = nn.Conv2d(in_channels=decoder_channels, out_channels=1, kernel_size=3, padding=1)
    
    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_out = self.backbone.decoder(features)
        seg_logits = self.backbone.segmentation_head(decoder_out)
        return seg_logits

# 2. DeepLabV3+
class MultiTaskDeepLab(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="tu-convnext_base", 
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
        try:
            if hasattr(self.backbone, 'segmentation_head') and self.backbone.segmentation_head is not None:
                first_layer = self.backbone.segmentation_head[0]
                if isinstance(first_layer, nn.Conv2d):
                    decoder_channels = first_layer.in_channels
                else:
                     decoder_channels = 256
            else:
                 decoder_channels = 256
        except Exception:
             decoder_channels = 256
        self.depth_head = nn.Conv2d(in_channels=decoder_channels, out_channels=1, kernel_size=3, padding=1)
    
    def forward(self, x):
        features = self.backbone.encoder(x)
        decoder_output = self.backbone.decoder(*features)
        seg_logits = self.backbone.segmentation_head(decoder_output)
        return seg_logits

# --- Utils ---
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_miou(cm):
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = intersection / union
    miou = np.nanmean(iou)
    return miou

def update_cm(preds, labels, num_classes, ignore_index, cm):
    preds = preds.flatten()
    labels = labels.flatten()
    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]
    add_cm = np.bincount(
        num_classes * labels + preds,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    cm += add_cm
    return cm

# --- Dataset and TTA Logic ---
class EnsembleDataset(Dataset):
    def __init__(self, image_paths, label_paths, depth_paths, cfg):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.depth_paths = depth_paths
        self.cfg = cfg
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load Raw Image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # [H, W, 3]
        
        # Load Label
        label = cv2.imread(self.label_paths[idx], cv2.IMREAD_UNCHANGED)
        if label.ndim == 3: label = label[:, :, 0]
        
        # Prepare Depth Input (for 093.5)
        depth_input = None
        if self.depth_paths is not None:
            raw_depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_UNCHANGED)
            raw_depth = raw_depth.astype(np.float32) / 1000.0
            dmin, dmax = self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX
            raw_depth = np.clip(raw_depth, dmin, dmax)
            # Inverse Depth Norm
            inv_d = 1.0 / raw_depth
            inv_min = 1.0 / dmax
            inv_max = 1.0 / dmin
            norm_inv_d = (inv_d - inv_min) / (inv_max - inv_min)
            depth_input = np.clip(norm_inv_d, 0.0, 1.0).astype(np.float32) # [H, W]

        # Consistency: Resize Image/Depth to Base Resolution (matches Training)
        # Note: Label is kept at ORIGINAL resolution for evaluation
        img_resized = cv2.resize(image, (self.cfg.BASE_WIDTH, self.cfg.BASE_HEIGHT), interpolation=cv2.INTER_LINEAR)
        
        depth_input_resized = None
        if depth_input is not None:
            depth_input_resized = cv2.resize(depth_input, (self.cfg.BASE_WIDTH, self.cfg.BASE_HEIGHT), interpolation=cv2.INTER_LINEAR)

        return img_resized, label, depth_input_resized

def tta_inference_ensemble(models, image, depth_input, cfg):
    """
    Run TTA inference.
    Input image/depth should be at BASE resolution (e.g. 576x768).
    """
    h_base, w_base = image.shape[:2]
    
    mean_tensor = torch.tensor(cfg.MEAN).view(1, 1, 3).to(cfg.DEVICE)
    std_tensor = torch.tensor(cfg.STD).view(1, 1, 3).to(cfg.DEVICE)
    
    model_accumulators = [None] * len(models)
    count = 0
    
    for scale, flip in cfg.TTA_COMBS:
        # Scale relative to BASE resolution
        h_new = int(h_base * scale)
        w_new = int(w_base * scale)
        h_new = (h_new // 32) * 32
        w_new = (w_new // 32) * 32
        
        img_aug = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        if depth_input is not None:
            depth_aug = cv2.resize(depth_input, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        
        if flip:
            img_aug = cv2.flip(img_aug, 1)
            if depth_input is not None:
                depth_aug = cv2.flip(depth_aug, 1)
                
        img_tensor_base = torch.from_numpy(img_aug).float().to(cfg.DEVICE) / 255.0
        img_tensor_base = (img_tensor_base - mean_tensor) / std_tensor 
        
        for i, model_info in enumerate(models):
            net = model_info['model']
            in_ch = model_info['in_channels']
            
            if in_ch == 4:
                if depth_input is None:
                    raise ValueError("Model expects 4 channels but depth input is None")
                d_tensor = torch.from_numpy(depth_aug).float().to(cfg.DEVICE).unsqueeze(2) 
                input_tensor = torch.cat([img_tensor_base, d_tensor], dim=2)
            else:
                input_tensor = img_tensor_base
                
            input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
            
            with torch.no_grad():
                logits = net(input_tensor) 
                probs = F.softmax(logits / cfg.INFERENCE_TEMP, dim=1)
                
            probs = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            if flip: probs = cv2.flip(probs, 1)
            # Resize back to BASE resolution
            probs = cv2.resize(probs, (w_base, h_base), interpolation=cv2.INTER_LINEAR)
            
            if model_accumulators[i] is None:
                model_accumulators[i] = probs
            else:
                model_accumulators[i] += probs
        
        count += 1
        
    avg_probs_per_model = [acc / count for acc in model_accumulators]
    return avg_probs_per_model

# --- Main Logic ---
def main():
    cfg = Config()
    seed_everything(cfg.SEED)
    
    # Check Checkpoints first
    print("Checking model checkpoints...")
    all_ckpts_exist = True
    for m_cfg in cfg.MODELS:
        for fold in range(cfg.N_FOLDS):
            p = m_cfg['path_template'].format(fold)
            if not os.path.exists(p) and os.path.exists('/Users/koizumihinata/NYUv2/' + p):
                p = '/Users/koizumihinata/NYUv2/' + p
                
            if not os.path.exists(p):
                print(f"[Error] Checkpoint missing: {p}")
                all_ckpts_exist = False
    
    if not all_ckpts_exist:
        print("Stopping due to missing checkpoints.")
        return

    os.makedirs(os.path.join("data/outputs"), exist_ok=True)

    # Fix Data Paths
    if not os.path.exists(cfg.DATA_ROOT) and os.path.exists('/Users/koizumihinata/NYUv2/' + cfg.DATA_ROOT):
        p = '/Users/koizumihinata/NYUv2/'
        cfg.DATA_ROOT = p + cfg.DATA_ROOT
        for m in cfg.MODELS:
            m['path_template'] = p + m['path_template']

    image_dir = os.path.join(cfg.DATA_ROOT, 'image')
    label_dir = os.path.join(cfg.DATA_ROOT, 'label')
    depth_dir = os.path.join(cfg.DATA_ROOT, 'depth')
    
    image_files = sorted(os.listdir(image_dir))
    label_files = sorted(os.listdir(label_dir))
    depth_files = sorted(os.listdir(depth_dir))
    
    image_paths = np.array([os.path.join(image_dir, f) for f in image_files])
    label_paths = np.array([os.path.join(label_dir, f) for f in label_files])
    depth_paths = np.array([os.path.join(depth_dir, f) for f in depth_files])
    
    kf = KFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    fold_splits = list(kf.split(image_paths))
    
    oof_data = [] 
    
    print("Starting OOF Inference for Ensemble...")
    for fold_idx in range(cfg.N_FOLDS):
        print(f"\nFold {fold_idx} / {cfg.N_FOLDS - 1}")
        _, valid_idx = fold_splits[fold_idx]
        
        current_models = []
        for m_cfg in cfg.MODELS:
            ckpt_path = m_cfg['path_template'].format(fold_idx)
            if m_cfg['type'] == 'FPN':
                net = MultiTaskFPN(cfg.NUM_CLASSES, m_cfg['in_channels'])
            else:
                net = MultiTaskDeepLab(cfg.NUM_CLASSES, m_cfg['in_channels'])
                
            state = torch.load(ckpt_path, map_location=cfg.DEVICE)
            net.load_state_dict(state, strict=False)
            net.to(cfg.DEVICE)
            net.eval()
            current_models.append({'model': net, 'in_channels': m_cfg['in_channels']})
            
        dataset = EnsembleDataset(image_paths[valid_idx], label_paths[valid_idx], depth_paths[valid_idx], cfg)
        
        for i in tqdm(range(len(dataset))):
            # img is Resized (576x768), label is Original
            img, label, d_input = dataset[i]
            
            probs_list = tta_inference_ensemble(current_models, img, d_input, cfg)
            
            # probs_list is at BASE resolution (576x768)
            # Resize to small storage size to save RAM
            h_small = int(cfg.BASE_HEIGHT * cfg.STORE_SCALE)
            w_small = int(cfg.BASE_WIDTH * cfg.STORE_SCALE)
            
            probs_list_small = []
            for p in probs_list:
                p_small = cv2.resize(p, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
                probs_list_small.append(p_small.astype(np.float16))
            
            oof_data.append({
                'label': label, # Original Resolution
                'probs': probs_list_small,
                'orig_size': label.shape[:2] # (H, W)
            })
            
    print(f"\nCollected {len(oof_data)} OOF samples. Starting Optimization...")
    
    def calculate_oof_miou(weights):
        # Constraints handle sum=1, so we don't normalize here.
        w = np.array(weights)
        
        cm = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES), dtype=np.int64)
        
        for sample in oof_data:
            lbl = sample['label']
            probs_list = sample['probs'] # Small size
            orig_h, orig_w = sample['orig_size']
            
            ensemble_prob_small = np.zeros_like(probs_list[0], dtype=np.float32)
            for i, p in enumerate(probs_list):
                ensemble_prob_small += w[i] * p.astype(np.float32)
                
            # Resize back to ORIGINAL resolution for evaluation
            ensemble_prob = cv2.resize(ensemble_prob_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            pred = np.argmax(ensemble_prob, axis=2)
            cm = update_cm(pred, lbl, cfg.NUM_CLASSES, cfg.IGNORE_INDEX, cm)
            
        return -compute_miou(cm)
    
    n_models = len(cfg.MODELS)
    
    # 0. Single Model Scores
    print("Calculating Single Model Scores...")
    single_scores = {}
    for i, m in enumerate(cfg.MODELS):
        w_single = [0.0] * n_models
        w_single[i] = 1.0
        score = -calculate_oof_miou(w_single)
        print(f"  {m['name']}: {score:.5f}")
        single_scores[m['name']] = score

    # 1. Equal Weights
    init_w = [1.0/n_models] * n_models
    print("Calculating Equal Weight Score...")
    score_eq = -calculate_oof_miou(init_w)
    print(f"Equal Weights {init_w}: mIoU = {score_eq:.5f}")
    
    # 2. Manual Weights (2 models)
    manual_w = [0.5, 0.5]
    print("Calculating Manual Weight Score...")
    score_man = -calculate_oof_miou(manual_w)
    print(f"Manual Weights {manual_w}: mIoU = {score_man:.5f}")
    
    # 3. Optimization
    print("Optimizing Weights...")
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    bounds = tuple((0, 1) for _ in range(n_models))
    res = minimize(calculate_oof_miou, init_w, method='SLSQP', bounds=bounds, constraints=cons, tol=1e-4)
    
    best_w = res.x / res.x.sum() # Final normalization for output safety
    best_score = -res.fun
    
    print("\noptimization Complete!")
    print(f"Best mIoU: {best_score:.5f}")
    print("Best Weights:")
    for i, m in enumerate(cfg.MODELS):
        print(f"  {m['name']}: {best_w[i]:.4f}")
        
    results = {
        'single_models': single_scores,
        'equal_weights': {'w': init_w, 'miou': score_eq},
        'manual_weights': {'w': manual_w, 'miou': score_man},
        'optimized_weights': {'w': best_w.tolist(), 'miou': best_score}
    }
    
    with open(os.path.join("data/outputs", f"{cfg.EXP_NAME}_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Test Inference with Optimized Weights
    print("\n" + "="*50)
    print("Starting Test Inference with Optimized Weights...")
    print("="*50)
    
    test_image_dir = os.path.join("data/test", 'image')
    test_depth_dir = os.path.join("data/test", 'depth')
    
    if not os.path.exists(test_image_dir):
        print(f"Test image directory not found: {test_image_dir}")
        return
    
    test_image_files = sorted([f for f in os.listdir(test_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    test_image_paths = [os.path.join(test_image_dir, f) for f in test_image_files]
    
    test_depth_paths = None
    if os.path.exists(test_depth_dir):
        test_depth_files = sorted([f for f in os.listdir(test_depth_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        test_depth_paths = [os.path.join(test_depth_dir, f) for f in test_depth_files]
    
    print(f"Found {len(test_image_paths)} test images")
    
    # Load all models for all folds
    all_fold_models = []
    for fold_idx in range(cfg.N_FOLDS):
        fold_models = []
        for m_cfg in cfg.MODELS:
            ckpt_path = m_cfg['path_template'].format(fold_idx)
            if m_cfg['type'] == 'FPN':
                net = MultiTaskFPN(cfg.NUM_CLASSES, m_cfg['in_channels'])
            else:
                net = MultiTaskDeepLab(cfg.NUM_CLASSES, m_cfg['in_channels'])
            state = torch.load(ckpt_path, map_location=cfg.DEVICE)
            net.load_state_dict(state, strict=False)
            net.to(cfg.DEVICE)
            net.eval()
            fold_models.append({'model': net, 'in_channels': m_cfg['in_channels']})
        all_fold_models.append(fold_models)
    
    # Test dataset (without labels)
    class TestDataset(Dataset):
        def __init__(self, image_paths, depth_paths, cfg):
            self.image_paths = image_paths
            self.depth_paths = depth_paths
            self.cfg = cfg
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image = cv2.imread(self.image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(image, (self.cfg.BASE_WIDTH, self.cfg.BASE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            depth_input_resized = None
            if self.depth_paths is not None and idx < len(self.depth_paths):
                raw_depth = cv2.imread(self.depth_paths[idx], cv2.IMREAD_UNCHANGED)
                raw_depth = raw_depth.astype(np.float32) / 1000.0
                dmin, dmax = self.cfg.DEPTH_MIN, self.cfg.DEPTH_MAX
                raw_depth = np.clip(raw_depth, dmin, dmax)
                inv_d = 1.0 / raw_depth
                inv_min = 1.0 / dmax
                inv_max = 1.0 / dmin
                norm_inv_d = (inv_d - inv_min) / (inv_max - inv_min)
                depth_input = np.clip(norm_inv_d, 0.0, 1.0).astype(np.float32)
                depth_input_resized = cv2.resize(depth_input, (self.cfg.BASE_WIDTH, self.cfg.BASE_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
            return img_resized, depth_input_resized
    
    test_dataset = TestDataset(test_image_paths, test_depth_paths, cfg)
    all_test_preds = []
    
    # Use optimized weights
    ensemble_weights = np.array(best_w)
    
    print(f"Using optimized weights: {ensemble_weights}")
    
    for i in tqdm(range(len(test_dataset)), desc="Test Inference"):
        img, d_input = test_dataset[i]
        
        # Ensemble across all folds
        fold_probs_list = []
        for fold_idx, fold_models in enumerate(all_fold_models):
            probs_list = tta_inference_ensemble(fold_models, img, d_input, cfg)
            fold_probs_list.append(probs_list)
        
        # Average across folds, then weighted ensemble across models
        final_probs = None
        for model_idx in range(len(cfg.MODELS)):
            model_probs = None
            for fold_idx in range(cfg.N_FOLDS):
                if model_probs is None:
                    model_probs = fold_probs_list[fold_idx][model_idx]
                else:
                    model_probs += fold_probs_list[fold_idx][model_idx]
            model_probs /= cfg.N_FOLDS
            
            if final_probs is None:
                final_probs = ensemble_weights[model_idx] * model_probs
            else:
                final_probs += ensemble_weights[model_idx] * model_probs
        
        # Get prediction at BASE resolution
        pred = np.argmax(final_probs, axis=2).astype(np.uint8)
        all_test_preds.append(pred)
    
    # Stack all predictions: [N, H, W]
    submission = np.stack(all_test_preds, axis=0)
    
    # Save as .npy
    output_path = os.path.join("data/outputs", f"{cfg.EXP_NAME}_submission.npy")
    np.save(output_path, submission)
    print(f"\nSaved test predictions: {output_path}")
    print(f"Shape: {submission.shape} (N={submission.shape[0]}, H={submission.shape[1]}, W={submission.shape[2]})")

if __name__ == '__main__':
    main()
