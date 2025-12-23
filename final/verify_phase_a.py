
import sys
import os
import torch

# Add final directory to path to import modules
sys.path.append('/Users/koizumihinata/NYUv2/final')

def verify_exp100_1():
    print("Verifying exp100_1_deeplab...")
    try:
        from exp100_1_deeplab import MultiTaskDeepLabV3Plus, Config
        model = MultiTaskDeepLabV3Plus(num_classes=Config.NUM_CLASSES, in_channels=4)
        print("Model initialized successfully.")
        
        # Test forward pass with dummy input
        x = torch.randn(2, 4, 576, 768)
        logits, depth = model(x)
        print(f"Forward pass successful. Output shapes: Logits {logits.shape}, Depth {depth.shape}")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

def verify_exp100_2():
    print("\nVerifying exp100_2_pspnet...")
    try:
        from exp100_2_pspnet import MultiTaskPSPNet, Config
        model = MultiTaskPSPNet(num_classes=Config.NUM_CLASSES, in_channels=4)
        print("Model initialized successfully.")
        
        # Test forward pass with dummy input
        x = torch.randn(2, 4, 576, 768)
        logits, depth = model(x)
        print(f"Forward pass successful. Output shapes: Logits {logits.shape}, Depth {depth.shape}")
        
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_exp100_1()
    verify_exp100_2()
