import torch
import torch.nn as nn
from model.segformer import segformer_mit_b3

def verify_model():
    print("Initializing SegFormer MiT-B3...")
    # Initialize with pretrained=False to avoid downloading if not cached (structure check only)
    model = segformer_mit_b3(num_classes=13, in_channels=4, pretrained=False)
    
    print("\n--- Inspecting SegFormerHead Projection Layers ---")
    head = model.head
    print(f"linear_c4: {head.linear_c4}")
    print(f"linear_c3: {head.linear_c3}")
    print(f"linear_c2: {head.linear_c2}")
    print(f"linear_c1: {head.linear_c1}")
    
    # Check if they are just Conv2d
    for name, layer in [('linear_c4', head.linear_c4), ('linear_c3', head.linear_c3), 
                        ('linear_c2', head.linear_c2), ('linear_c1', head.linear_c1)]:
        if isinstance(layer, nn.Conv2d):
            print(f"[PASS] {name} is nn.Conv2d")
        else:
            print(f"[FAIL] {name} is {type(layer)}, expected nn.Conv2d")

    print("\n--- Running Dummy Forward Pass ---")
    x = torch.randn(1, 4, 640, 640)
    try:
        y = model(x)
        print(f"[PASS] Forward pass successful. Output shape: {y.shape}")
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")

if __name__ == "__main__":
    verify_model()
