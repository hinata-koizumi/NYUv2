import torch
import numpy as np
import cv2
import albumentations as A
import unittest
from src.engine.inference import Predictor

# Mock Config
class MockConfig:
    RESIZE_HEIGHT = 720
    RESIZE_WIDTH = 960
    NUM_CLASSES = 13
    IGNORE_INDEX = 255
    TTA_COMBS = [(1.0, False), (1.0, True)] # Scale 1.0, Flip/NoFlip

class TestPipeline(unittest.TestCase):
    def test_pad_unpad(self):
        """
        Verify that our specific Resize->Pad logic in transforms 
        matches the Unpad logic in inference using 'meta'.
        """
        cfg = MockConfig()
        
        # 1. Create Dummy Image similar to NYUv2 (480x640)
        orig_h, orig_w = 480, 640
        img = np.random.randint(0, 255, (orig_h, orig_w, 3), dtype=np.uint8)
        
        # 2. Apply Valid Transform (Resize 720x960 -> Pad to mult 32)
        # 720 is mult of 32? 720/32 = 22.5 -> Pad to 736
        # 960 is mult of 32? 960/32 = 30 -> No Pad
        # So Pad should be (736, 960)
        
        # Re-implement transform logic to match expected behavior
        h_pad_target = ((cfg.RESIZE_HEIGHT + 31) // 32) * 32
        w_pad_target = ((cfg.RESIZE_WIDTH + 31) // 32) * 32
        
        self.assertEqual(h_pad_target, 736)
        self.assertEqual(w_pad_target, 960)
        
        transform = A.Compose([
            A.Resize(height=cfg.RESIZE_HEIGHT, width=cfg.RESIZE_WIDTH),
            A.PadIfNeeded(
                min_height=h_pad_target, 
                min_width=w_pad_target, 
                border_mode=cv2.BORDER_CONSTANT, 
                value=0, 
                position="top_left"
            )
        ])
        
        t = transform(image=img)
        img_trans = t["image"]
        
        # Check Transformed Size
        self.assertEqual(img_trans.shape[0], 736)
        self.assertEqual(img_trans.shape[1], 960)
        
        # Check Content Location (Top-Left)
        # Content should be 720x960. 
        # Padding should be separate.
        # Let's verify pixel content preservation after Resize.
        # But Resize is lossy.
        # Let's check Padding zeros.
        # Content is 0:720. 720:736 should be 0.
        pad_area = img_trans[720:, :, :]
        self.assertTrue((pad_area == 0).all(), "Padding area is not zero")
        
        # 3. Test Unpadding Logic
        # Construct Meta
        meta = {
            "orig_h": torch.tensor([orig_h]),
            "orig_w": torch.tensor([orig_w]),
            "pad_h": torch.tensor([736 - 720]), # 16
            "pad_w": torch.tensor([960 - 960]), # 0
        }
        
        # Mock Logits (C, H_pad, W_pad)
        # Create a pattern we can verify.
        # e.g. Valid area = 1, Pad area = 0
        logits = torch.ones((13, 736, 960), dtype=torch.float32)
        logits[:, 720:, :] = 0 # Pad area
        
        # Predictor._unpad_and_resize (we instantiate wrapper)
        class MockPredictor(Predictor):
            def __init__(self): pass
            
        p = MockPredictor()
        out = p._unpad_and_resize(logits, {k: v[0] for k, v in meta.items()})
        
        # Result should be (C, orig_h, orig_w)
        self.assertEqual(out.shape, (13, orig_h, orig_w))
        
        # Verification:
        # Unpad crops 0:720.
        # Then Resize (Bilinear) 720x960 -> 480x640.
        # Since input was all 1s in valid area, output should be all 1s (approx).
        self.assertTrue(torch.allclose(out, torch.ones_like(out), atol=1e-5), "Unpad+Resize corrupted content")

    def test_tta_flip_invert(self):
        """
        Verify TTA Flip -> Invert Flip.
        """
        # (B, C, H, W)
        x = torch.randn(1, 13, 100, 100)
        
        # Flip
        x_g = torch.flip(x, dims=[3])
        
        # Invert
        x_rec = torch.flip(x_g, dims=[3])
        
        self.assertTrue(torch.equal(x, x_rec), "Flip inversion failed")
        
    def test_tta_scale(self):
        # Scale -> Resize Back
        # This is lossy, but we check shapes.
        x = torch.randn(1, 13, 100, 100)
        scale = 1.5
        h_new = int(100 * scale)
        w_new = int(100 * scale)
        
        x_s = torch.nn.functional.interpolate(x, size=(h_new, w_new), mode='bilinear')
        x_rec = torch.nn.functional.interpolate(x_s, size=(100, 100), mode='bilinear')
        
        self.assertEqual(x_rec.shape, x.shape)

if __name__ == '__main__':
    unittest.main()
