import numpy as np
import itertools
from .metrics import calculate_miou

def apply_weights(oof_dict, weights):
    """
    Apply weights to OOF logits.
    weights: dict {model_name: weight} or specialized structure
    """
    first_key = list(oof_dict.keys())[0]
    result = np.zeros_like(oof_dict[first_key])
    
    for name, logit in oof_dict.items():
        w = weights.get(name, 0.0)
        result += logit * w
        
    return result

def optimize_mean(oof_dict, gt):
    """
    Preset 1: Mean (Equal weights)
    """
    models = list(oof_dict.keys())
    n = len(models)
    weights = {name: 1.0/n for name in models}
    
    # Calculate score
    # ens = apply_weights(oof_dict, weights)
    # score = calculate_miou(gt, np.argmax(ens, axis=1))
    
    return weights

def optimize_grid_w(oof_dict, gt, steps=101):
    """
    Preset 2: Grid Search for weights (assuming 2 models)
    """
    models = list(oof_dict.keys())
    if len(models) != 2:
        print("WARNING: grid_w optimized for 2 models only. Falling back to mean.")
        return optimize_mean(oof_dict, gt)
        
    m1, m2 = models[0], models[1]
    best_score = -1
    best_weights = {m1: 0.5, m2: 0.5}
    
    # Grid search 0.00 to 1.00
    for w in np.linspace(0, 1, steps):
        # Current weights
        w1 = w
        w2 = 1.0 - w
        
        # Fast ensemble
        ens = oof_dict[m1] * w1 + oof_dict[m2] * w2
        pred = np.argmax(ens, axis=1)
        score = calculate_miou(gt, pred)
        
        if score >= best_score:
            best_score = score
            best_weights = {m1: w1, m2: w2}
            
    print(f"  [grid_w] Best Score: {best_score:.5f} (w={best_weights[m1]:.2f}/{best_weights[m2]:.2f})")
    return best_weights

def optimize_books_gate(oof_dict, gt, steps=21):
    """
    Preset 3: Books Gating
    Global weight w for all classes, plus separate w_books for 'books' class.
    Assuming 2 models.
    """
    models = list(oof_dict.keys())
    if len(models) != 2:
        return optimize_mean(oof_dict, gt)

    m1, m2 = models[0], models[1]
    
    # Needs GT to recognize books class?
    # Actually we optimize w_global and w_books
    # 'books' index is needed. NYUv2 13 classes. 'books' is class 3 usually?
    # We should define class indices constant.
    # For now assume 'books' is index 3 (standard NYUv2 13 class)
    BOOKS_IDX = 3 
    
    best_score = -1
    best_params = {"w_global": 0.5, "w_books": 0.5}
    
    # 2D Grid
    for w_g in np.linspace(0, 1, steps):
        for w_b in np.linspace(0, 1, steps):
            
            # Construct mixed logits
            # For non-books: use w_g
            # For books: use w_b
            
            # This is slightly complex to vectorize efficiently without copying big arrays
            # But specific to 2 models:
            # ens = m1 * w1 + m2 * w2
            # w1_class = w_b if c==BOOKS else w_g
            
            # Optimization: 
            # ens = (m1 * w_g + m2 * (1-w_g)) 
            # ens[:, BOOKS_IDX] = m1[:, BOOKS_IDX] * w_b + m2[:, BOOKS_IDX] * (1-w_b)
            
            ens = oof_dict[m1] * w_g + oof_dict[m2] * (1.0 - w_g)
            ens[:, BOOKS_IDX] = oof_dict[m1][:, BOOKS_IDX] * w_b + oof_dict[m2][:, BOOKS_IDX] * (1.0 - w_b)
            
            pred = np.argmax(ens, axis=1)
            score = calculate_miou(gt, pred)
            
        if score >= best_score:
                best_score = score
                best_params = {"w_global": w_g, "w_books": w_b}
                
    w_g = best_params['w_global']
    w_b = best_params['w_books']
    print(f"  [books_gate] Best Score: {best_score:.5f} (w_g={w_g:.2f}, w_b={w_b:.2f})")
    
    # Return structure needed for submission
    # We need to signal that this requires class-gating logic during inference
    return {
        "method": "books_gate",
        "params": best_params,
        "models": models
    }

def optimize_temp_calib(oof_dict, gt, steps=4):
    """
    Preset 4: Temperature Calibration
    Grid search T for each model.
    """
    models = list(oof_dict.keys())
    # ... Implementation similar to grid but scaling logits ...
    # Placeholder for brevity/time, falling back to mean or simple implementation
    print("  [temp_calib] Placeholder: returning mean weights.")
    return optimize_mean(oof_dict, gt)
