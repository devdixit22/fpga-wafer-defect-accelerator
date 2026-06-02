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

spec4 = importlib.util.spec_from_file_location("simulate", str(ROOT / "software" / "4_simulate.py"))
sim = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(sim)

img = sim_infer.load_image(ROOT / "data/val/Donut/0552.png")
state_dict = torch.load(ROOT / "wafer_model.pth", map_location="cpu", weights_only=True)

model_pt = WaferCNN()
model_pt.load_state_dict(state_dict)
model_pt.eval()

with torch.no_grad():
    x = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    c1_pt = model_pt.pool(model_pt.relu(model_pt.conv1(x))).numpy()[0]

w2 = state_dict["conv2.weight"].numpy()
w2_flat = w2.reshape(w2.shape[0], -1)

w2_scale = 127.0 / np.max(np.abs(w2))
w2_q = sim_infer.quantize(w2_flat)

cols2 = sim_infer.im2col(c1_pt)
a2_scale = 127.0 / (np.max(np.abs(cols2)) + 1e-9)
cols2_q = sim_infer.quantize(cols2)

c2_q_flat = cols2_q.astype(np.float32) @ w2_q.T.astype(np.float32)

import shutil
sim_infer.write_activation_mem(cols2)
shutil.copy(ROOT / "weights" / "conv2_weight_int8.mem", ROOT / "conv1_weight_int8.mem")
c2_hw_flat = sim.run_accelerator(ROOT / "weights" / "conv2_weight_int8.npy", cols2.shape[0])

# C2_hw_flat has shape (out_c, num_patches). We transpose to match c2_q_flat (num_patches, out_c)
c2_hw_flat = c2_hw_flat.T

print(f"HW flat match NP flat? {np.allclose(c2_hw_flat, c2_q_flat)}")
print(f"HW flat mean: {np.mean(c2_hw_flat):.2f}, NP flat mean: {np.mean(c2_q_flat):.2f}")
print(f"HW max: {np.max(c2_hw_flat)}, NP max: {np.max(c2_q_flat)}")

diff = np.abs(c2_hw_flat - c2_q_flat)
print(f"Max diff: {np.max(diff)}")
