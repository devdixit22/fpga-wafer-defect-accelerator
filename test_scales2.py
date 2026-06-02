import torch
import numpy as np
import sys
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT))

import importlib.util
spec = importlib.util.spec_from_file_location("infer", str(ROOT / "software" / "5_infer.py"))
sim_infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim_infer)
load_image = sim_infer.load_image
im2col = sim_infer.im2col
quantize = sim_infer.quantize

spec2 = importlib.util.spec_from_file_location("train", str(ROOT / "software" / "2_train.py"))
sim_train = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(sim_train)
WaferCNN = sim_train.WaferCNN

from config import N, IMAGE_SIZE

img = load_image(ROOT / "test_image.png")
state_dict = torch.load(ROOT / "wafer_model.pth", map_location="cpu", weights_only=True)

model = WaferCNN()
model.load_state_dict(state_dict)
model.eval()

# PyTorch C1
with torch.no_grad():
    x = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    pt_c1 = model.conv1(x).numpy()[0]

# NumPy C1 (unquantized)
w1 = state_dict["conv1.weight"].numpy()
b1 = state_dict["conv1.bias"].numpy()
w1_flat = w1.reshape(w1.shape[0], -1)

cols1 = im2col(img)
np_c1_flat = cols1 @ w1_flat.T
np_c1 = np_c1_flat.T.reshape(N, IMAGE_SIZE, IMAGE_SIZE) + b1[:, None, None]

print(f"NP Float match PT? {np.allclose(np_c1, pt_c1, atol=1e-4)}")
print(f"NP C1 mean: {np.mean(np_c1):.4f}, PT C1 mean: {np.mean(pt_c1):.4f}")

# NumPy C1 (quantized)
w1_scale = 127.0 / np.max(np.abs(w1))
w1_q = quantize(w1_flat)

a1_scale = 127.0 / (np.max(np.abs(cols1)) + 1e-9)
cols1_q = quantize(cols1)

c1_q_flat = cols1_q.astype(np.float32) @ w1_q.T.astype(np.float32)
c1_q_deq = c1_q_flat / (w1_scale * a1_scale)
c1_q_deq = c1_q_deq.T.reshape(N, IMAGE_SIZE, IMAGE_SIZE) + b1[:, None, None]

print(f"NP Quant match PT? {np.allclose(c1_q_deq, pt_c1, atol=0.1)}")
print(f"NP Quant mean: {np.mean(c1_q_deq):.4f}, max: {np.max(c1_q_deq):.4f}")
