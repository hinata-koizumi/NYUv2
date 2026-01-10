import torch
import sys
import os
from huggingface_hub import hf_hub_download

# Add path
sys.path.append(os.getcwd())

from segformer.configs.base_config import Config
from segformer.model.meta_arch import build_model

def debug_keys():
    print("--- Debugging SegFormer Keys ---")
    cfg = Config()
    model = build_model(cfg)
    backbone = model.backbone
    
    # 1. Load HF Weights manually
    repo_id = "nvidia/mit-b3"
    print(f"Loading {repo_id}...")
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # 2. Simulate the mapping logic EXACTLY as in segformer.py
    # We copy-paste the logic here to reproduce the 'new_state_dict'
    new_state_dict = {}
    
    print("Mapping keys...")
    for k, v in state_dict.items():
        if not k.startswith("segformer.encoder"):
            continue
        
        new_k = k.replace("segformer.encoder.", "")
        
        # 1. Patch Embeddings
        new_k = new_k.replace("patch_embeddings.0", "patch_embed1")
        new_k = new_k.replace("patch_embeddings.1", "patch_embed2")
        new_k = new_k.replace("patch_embeddings.2", "patch_embed3")
        new_k = new_k.replace("patch_embeddings.3", "patch_embed4")
        
        # 2. Blocks
        new_k = new_k.replace("block.0.", "block1.")
        new_k = new_k.replace("block.1.", "block2.")
        new_k = new_k.replace("block.2.", "block3.")
        new_k = new_k.replace("block.3.", "block4.")
        
        # 3. LayerNorms & Internal Names
        # Block LayerNorms
        new_k = new_k.replace("layer_norm_1", "norm1")
        new_k = new_k.replace("layer_norm_2", "norm2")
        
        # Attention
        new_k = new_k.replace("attention.self.query", "attn.q")
        new_k = new_k.replace("attention.self.sr", "attn.sr")
        new_k = new_k.replace("attention.self.layer_norm", "attn.norm")
        new_k = new_k.replace("attention.output.dense", "attn.proj")
        
        # MLP
        new_k = new_k.replace("mlp.dense1", "mlp.fc1")
        new_k = new_k.replace("mlp.dense2", "mlp.fc2")
        new_k = new_k.replace("mlp.dwconv.dwconv", "mlp.dwconv")

        # Stage Norms (layer_norm.0 -> norm1)
        if new_k.startswith("layer_norm."):
             idx = int(new_k.split(".")[1])
             new_k = f"norm{idx+1}." + ".".join(new_k.split(".")[2:]) 

        # PatchEmbed LayerNorm (Remaining 'layer_norm' -> 'norm')
        new_k = new_k.replace("layer_norm", "norm")
        
        # Handle KV concatenation
        if "attention.self.key.weight" in k:
            v_val = state_dict[k.replace("key.weight", "value.weight")]
            new_k = new_k.replace("attention.self.key", "attn.kv")
            v = torch.cat([v, v_val], dim=0)
        elif "attention.self.key.bias" in k:
            v_val = state_dict[k.replace("key.bias", "value.bias")]
            new_k = new_k.replace("attention.self.key", "attn.kv")
            v = torch.cat([v, v_val], dim=0)
        elif "attention.self.value" in k:
            continue
        
        # Check against model
        if new_k in backbone.state_dict():
            model_shape = backbone.state_dict()[new_k].shape
            loaded_shape = v.shape
            
            # Shape adjustment logic
            if "mlp.fc" in new_k and "weight" in new_k:
                if v.dim() == 2:
                    v = v.unsqueeze(-1).unsqueeze(-1)
                    loaded_shape = v.shape # Update loaded shape
            
            if model_shape != loaded_shape:
                 print(f"SHAPE MISMATCH! {new_k}: Model {model_shape} vs Loaded {loaded_shape}")
            
            new_state_dict[new_k] = v
        else:
            # Key not found in model?
            # Check if it SHOULD be there
            if "block" in new_k:
                 pass # print(f"Unmapped/Unused HF Key: {k} -> {new_k}")

    # 3. Check what's missing in Backbone
    print("\n--- Missing Keys Analysis ---")
    backbone_keys = set(backbone.state_dict().keys())
    loaded_keys = set(new_state_dict.keys())
    missing = backbone_keys - loaded_keys
    print(f"Total Backbone Keys: {len(backbone_keys)}")
    print(f"Loaded Keys: {len(loaded_keys)}")
    print(f"Missing Keys: {len(missing)}")
    
    if len(missing) > 0:
        print("Sample missing keys:")
        for k in sorted(list(missing))[:20]:
            print(f"  {k}")

    # Check specifically for attn.proj
    print("\nChecking block1.1.attn.proj.weight...")
    if "block1.1.attn.proj.weight" in missing:
         print("  It is MISSING.")
         # Determine why
         # Was it mapped?
         # Reverse engineer: what HF key maps to this?
         # NVIDIA: segformer.encoder.block.0.1.attention.output.dense.weight
         target_hf_key = "segformer.encoder.block.0.1.attention.output.dense.weight"
         if target_hf_key in state_dict:
             print(f"  HF Key {target_hf_key} EXISTS.")
             v = state_dict[target_hf_key]
             print(f"  HF Shape: {v.shape}")
             print("  Mapping check:")
             # Run mapping
             nk = target_hf_key.replace("segformer.encoder.", "")
             nk = nk.replace("block.0.", "block1.")
             nk = nk.replace("attention.output.dense", "attn.proj")
             print(f"  Mapped to: {nk}")
             if nk in backbone.state_dict():
                 print(f"  Match found in backbone? Yes. Shape: {backbone.state_dict()[nk].shape}")
             else:
                 print("  Match found in backbone? NO.")
         else:
             print("  HF Key DOES NOT EXIST. Possible naming variant?")
             
if __name__ == "__main__":
    debug_keys()
