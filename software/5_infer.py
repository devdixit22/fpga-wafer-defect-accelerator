"""
Step 5: Full Inference Pipeline.

Takes an input image and runs the CNN forward pass.
- Conv layers run on the simulated FPGA (via 4_simulate.py).
- FC layers run in pure software NumPy.

Usage:
    python software/5_infer.py <image_path>
"""
import numpy as np
import cv2
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import N, IMAGE_SIZE, CLASS_NAMES

# Import the simulator bridge
sys.path.insert(0, str(ROOT / "software"))
# We use importlib because python modules starting with numbers are tricky to import normally
import importlib
sim = importlib.import_script = importlib.import_module("4_simulate")


# --- Preprocessing Helpers ---
def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None: raise FileNotFoundError(f"Cannot read {path}")
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img.astype(np.float32) / 255.0

def im2col(img, k=3, pad=1):
    if img.ndim == 2: img = np.expand_dims(img, 0)
    if pad > 0: img = np.pad(img, ((0,0),(pad,pad),(pad,pad)), mode='constant')
    C, H, W = img.shape
    out_h, out_w = H - k + 1, W - k + 1
    cols = []
    for y in range(out_h):
        for x in range(out_w):
            cols.append(img[:, y:y+k, x:x+k].flatten())
    return np.array(cols)

def quantize(x):
    scale = 127.0 / np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0
    return np.clip(np.round(x * scale), -128, 127).astype(np.int8)

def write_activation_mem(cols):
    num_p, in_f = cols.shape
    pad_in = (N - (in_f % N)) % N
    if pad_in > 0: cols = np.pad(cols, ((0,0),(0,pad_in)))
    in_blocks = cols.shape[1] // N
    q = quantize(cols).reshape(num_p, in_blocks, N)
    
    with open(ROOT / "activation.mem", "w") as f:
        for p in range(num_p):
            for ib in range(in_blocks):
                hex_str = "".join([f"{int(v) & 0xFF:02x}" for v in reversed(q[p, ib])])
                f.write(hex_str + "\n")


import torch

class FPGAInference:
    def __init__(self):
        state_dict = torch.load(ROOT / "wafer_model.pth", map_location="cpu", weights_only=True)
        self.w1 = state_dict["conv1.weight"].numpy()
        self.b1 = state_dict["conv1.bias"].numpy()
        self.w2 = state_dict["conv2.weight"].numpy()
        self.b2 = state_dict["conv2.bias"].numpy()
        self.w3 = state_dict["fc1.weight"].numpy()
        self.b3 = state_dict["fc1.bias"].numpy()
        self.w4 = state_dict["fc2.weight"].numpy()
        self.b4 = state_dict["fc2.bias"].numpy()
        
        self.w1_scale = 127.0 / np.max(np.abs(self.w1))
        self.w2_scale = 127.0 / np.max(np.abs(self.w2))

    def relu(self, x): return np.maximum(0, x)
    def maxpool(self, x):
        C, H, W = x.shape
        out = np.zeros((C, H//2, W//2), dtype=x.dtype)
        for y in range(H//2):
            for x_i in range(W//2):
                out[:, y, x_i] = np.max(x[:, y*2:y*2+2, x_i*2:x_i*2+2], axis=(1,2))
        return out

    def forward(self, img):
        print(f"Running inference (N={N}, {IMAGE_SIZE}x{IMAGE_SIZE})")
        
        # --- Conv1 on FPGA ---
        print("  Layer 1: CONV (Hardware)")
        cols1 = im2col(img)
        a1_scale = 127.0 / (np.max(np.abs(cols1)) + 1e-9)
        write_activation_mem(cols1)
        shutil.copy(ROOT / "weights" / "conv1_weight_int8.mem", ROOT / "conv1_weight_int8.mem")
        
        c1_int32 = sim.run_accelerator(ROOT / "weights" / "conv1_weight_int8.npy", cols1.shape[0])
        # DEQUANTIZE: c1_float = c1_int32 / (w_scale * a_scale)
        c1 = c1_int32 / (self.w1_scale * a1_scale)
        c1 = c1.reshape(N, IMAGE_SIZE, IMAGE_SIZE) + self.b1[:, None, None]
        c1 = self.maxpool(self.relu(c1))

        # --- Conv2 on FPGA ---
        print("  Layer 2: CONV (Hardware)")
        cols2 = im2col(c1)
        a2_scale = 127.0 / (np.max(np.abs(cols2)) + 1e-9)
        write_activation_mem(cols2)
        shutil.copy(ROOT / "weights" / "conv2_weight_int8.mem", ROOT / "conv1_weight_int8.mem")
        
        c2_int32 = sim.run_accelerator(ROOT / "weights" / "conv2_weight_int8.npy", cols2.shape[0])
        c2 = c2_int32 / (self.w2_scale * a2_scale)
        c2 = c2.reshape(N*2, IMAGE_SIZE//2, IMAGE_SIZE//2) + self.b2[:, None, None]
        c2 = self.maxpool(self.relu(c2))

        # --- FC layers in Software ---
        print("  Layer 3 & 4: FC (Software)")
        flat = c2.flatten()
        fc1 = self.relu(flat @ self.w3.T + self.b3)
        fc2 = fc1 @ self.w4.T + self.b4
        
        return CLASS_NAMES[np.argmax(fc2)]


def main():
    if len(sys.argv) < 2:
        print("Usage: python 5_infer.py <image_path>")
        sys.exit(1)
        
    img_path = Path(sys.argv[1])
    img = load_image(img_path)
    
    model = FPGAInference()
    pred = model.forward(img)
    print(f"\nResult: {pred}")


if __name__ == "__main__":
    main()
