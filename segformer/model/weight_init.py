
import torch
import torch.nn as nn
import math
from huggingface_hub import hf_hub_download

def init_weights_custom(m):
    """
    Custom weight initialization for SegFormer components.
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        nn.init.normal_(m.weight, std=math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()

def load_pretrained_mit_weights(backbone: nn.Module, phi: str) -> None:
    """
    Load pretrained MiT weights from NVIDIA's HuggingFace repo into the backbone.
    Handles key mapping between NVIDIA's structure and the local implementation.
    
    Args:
        backbone: The MixVisionTransformer backbone instance.
        phi: The variant identifier (e.g., 'b3').
    """
    repo_id = f"nvidia/mit-{phi}"
    print(f"Downloading/Loading pretrained weights from {repo_id}...")
    try:
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"[WARN] Failed to download weights from HF: {e}")
        return

    # Mapping: NVIDIA Official -> Our MixVisionTransformer
    # NVIDIA: segformer.encoder.patch_embeddings.0.proj.weight
    # Ours:   backbone.patch_embed1.proj.weight
    
    new_state_dict = {}
    matched_keys = 0
    
    for k, v in state_dict.items():
        if not k.startswith("segformer.encoder"):
            continue
        
        new_k = k.replace("segformer.encoder.", "")
        
        # 1. Patch Embeddings
        # patch_embeddings.0 -> patch_embed1
        new_k = new_k.replace("patch_embeddings.0", "patch_embed1")
        new_k = new_k.replace("patch_embeddings.1", "patch_embed2")
        new_k = new_k.replace("patch_embeddings.2", "patch_embed3")
        new_k = new_k.replace("patch_embeddings.3", "patch_embed4")
        
        # 2. Blocks
        # Use "block.0." to avoid matching "block.1.0" as "block.1.block1"
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
        new_k = new_k.replace("attention.self.layer_norm", "attn.norm") # This maps to Attention.norm
        new_k = new_k.replace("attention.output.dense", "attn.proj")
        
        # MLP
        new_k = new_k.replace("mlp.dense1", "mlp.fc1")
        new_k = new_k.replace("mlp.dense2", "mlp.fc2")
        new_k = new_k.replace("mlp.dwconv.dwconv", "mlp.dwconv")

        # Stage Norms (layer_norm.0 -> norm1)
        # Must happen BEFORE generic 'layer_norm' -> 'norm' replacement
        if new_k.startswith("layer_norm."):
            idx = int(new_k.split(".")[1])
            new_k = f"norm{idx+1}." + ".".join(new_k.split(".")[2:])
            
        # PatchEmbed LayerNorm (Remaining 'layer_norm' -> 'norm')
        new_k = new_k.replace("layer_norm", "norm") 
        
        # Handle KV concatenation (NVIDIA has key/value separate)
        if "attention.self.key.weight" in k:
            v_val = state_dict[k.replace("key.weight", "value.weight")]
            new_k = new_k.replace("attention.self.key", "attn.kv")
            v = torch.cat([v, v_val], dim=0)
        elif "attention.self.key.bias" in k:
            v_val = state_dict[k.replace("key.bias", "value.bias")]
            new_k = new_k.replace("attention.self.key", "attn.kv")
            v = torch.cat([v, v_val], dim=0)
        elif "attention.self.value" in k:
            continue # Already handled in key processing
        
        if new_k in backbone.state_dict():
            # Shape adjustment: NVIDIA (Linear) → Ours (Conv2d 1x1)
            if "mlp.fc" in new_k and "weight" in new_k:
                if v.dim() == 2:
                    v = v.unsqueeze(-1).unsqueeze(-1)
            
            # Handle 3-channel pretrained vs 4-channel model mismatch (RGB+Depth)
            if "patch_embed1.proj.weight" in new_k:
                current_shape = backbone.state_dict()[new_k].shape
                loaded_shape = v.shape
                # If model expects 4 channels but loaded weights have 3
                if current_shape[1] == 4 and loaded_shape[1] == 3:
                    print(f"Adapting {new_k} from {loaded_shape} to {current_shape} (RGB -> RGB+Depth Zero Init)")
                    new_v = torch.zeros(current_shape, dtype=v.dtype, device=v.device)
                    new_v[:, :3, :, :] = v  # Copy RGB
                    # 4th channel is already 0.0
                    v = new_v

            new_state_dict[new_k] = v
            matched_keys += 1
    
    msg = backbone.load_state_dict(new_state_dict, strict=False)
    print(f"Successfully loaded {matched_keys} keys into backbone.")
    print(f"Missing keys: {len(msg.missing_keys)}")
    if len(msg.missing_keys) < 20: 
         print(f"Missing sample: {msg.missing_keys[:5]}")
