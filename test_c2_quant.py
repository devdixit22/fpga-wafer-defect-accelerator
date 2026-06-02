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

spec2 = importlib.util.spec_from_file_location("train", str(ROOT / "software" / "2_train.py"))
sim_train = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(sim_train)
WaferCNN = sim_train.WaferCNN

img_path = ROOT / "data" / "val" / "Donut" / "Donut_0552.png"
if not img_path.exists():
    img_path = ROOT / "data" / "val" / "Donut" / "0552.png"

img = sim_infer.load_image(img_path)
state_dict = torch.load(ROOT / "wafer_model.pth", map_location="cpu", weights_only=True)

model_pt = WaferCNN()
model_pt.load_state_dict(state_dict)
model_pt.eval()

with torch.no_grad():
    x = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    c1_pt = model_pt.pool(model_pt.relu(model_pt.conv1(x))).numpy()[0]
    c2_pt = model_pt.pool(model_pt.relu(model_pt.conv2(torch.tensor(c1_pt).unsqueeze(0)))).numpy()[0]

w2 = state_dict["conv2.weight"].numpy()
b2 = state_dict["conv2.bias"].numpy()
w2_flat = w2.reshape(w2.shape[0], -1)

w2_scale = 127.0 / np.max(np.abs(w2))
w2_q = sim_infer.quantize(w2_flat)

cols2 = sim_infer.im2col(c1_pt)
a2_scale = 127.0 / (np.max(np.abs(cols2)) + 1e-9)
cols2_q = sim_infer.quantize(cols2)

c2_q_flat = cols2_q.astype(np.float32) @ w2_q.T.astype(np.float32)
c2_q_deq = c2_q_flat / (w2_scale * a2_scale)
c2_q_deq = c2_q_deq.T.reshape(16, 16, 16) + b2[:, None, None]
c2_q_deq = np.maximum(0, c2_q_deq)

# maxpool
C, H, W = c2_q_deq.shape
c2_pool = np.zeros((C, H//2, W//2))
for y in range(H//2):
    for x in range(W//2):
        c2_pool[:, y, x] = np.max(c2_q_deq[:, y*2:y*2+2, x*2:x*2+2], axis=(1,2))

print(f"NP Quant Match PT? {np.allclose(c2_pool, c2_pt, atol=1.0)}")
print(f"NP Quant mean: {np.mean(c2_pool):.4f}, PT mean: {np.mean(c2_pt):.4f}")
