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
FPGAInference = sim_infer.FPGAInference
load_image = sim_infer.load_image
im2col = sim_infer.im2col
quantize = sim_infer.quantize
write_activation_mem = sim_infer.write_activation_mem
spec2 = importlib.util.spec_from_file_location("train", str(ROOT / "software" / "2_train.py"))
sim_train = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(sim_train)
WaferCNN = sim_train.WaferCNN
spec4 = importlib.util.spec_from_file_location("simulate", str(ROOT / "software" / "4_simulate.py"))
sim = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(sim)
from config import N, IMAGE_SIZE

img = load_image(ROOT / "test_image.png")
state_dict = torch.load(ROOT / "wafer_model.pth", map_location="cpu", weights_only=True)

# 1. PyTorch Forward
model = WaferCNN()
model.load_state_dict(state_dict)
model.eval()
with torch.no_grad():
    pt_out = model(torch.tensor(img).unsqueeze(0).unsqueeze(0))
    print(f"PyTorch Pred: {pt_out.argmax(1).item()}")

# 2. Manual Quantized Inference
# Conv1
w1_f = state_dict["conv1.weight"].numpy()
w1_s = 127.0 / np.max(np.abs(w1_f))

cols1 = im2col(img)
a1_s = 127.0 / (np.max(np.abs(cols1)) + 1e-9)

import shutil
write_activation_mem(cols1)
shutil.copy(ROOT / "weights" / "conv1_weight_int8.mem", ROOT / "conv1_weight_int8.mem")
c1_int32 = sim.run_accelerator(ROOT / "weights" / "conv1_weight_int8.npy", cols1.shape[0])

# Dequantize
c1_float = c1_int32 / (w1_s * a1_s)
c1_float = c1_float.reshape(N, IMAGE_SIZE, IMAGE_SIZE) + state_dict["conv1.bias"].numpy()[:, None, None]
c1_float = np.maximum(0, c1_float) # relu
# maxpool
C, H, W = c1_float.shape
c1_pool = np.zeros((C, H//2, W//2))
for y in range(H//2):
    for x in range(W//2):
        c1_pool[:, y, x] = np.max(c1_float[:, y*2:y*2+2, x*2:x*2+2], axis=(1,2))

c1_pt = model.pool(model.relu(model.conv1(torch.tensor(img).unsqueeze(0).unsqueeze(0)))).detach().numpy()[0]
print(f"HW C1 match PT? {np.allclose(c1_pool, c1_pt, atol=1.0)}")
print(f"HW C1 mean: {np.mean(c1_pool):.4f}, max: {np.max(c1_pool):.4f}")
print(f"PT C1 mean: {np.mean(c1_pt):.4f}, max: {np.max(c1_pt):.4f}")
