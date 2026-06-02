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

spec2 = importlib.util.spec_from_file_location("train", str(ROOT / "software" / "2_train.py"))
sim_train = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(sim_train)
WaferCNN = sim_train.WaferCNN

img_path = ROOT / "data" / "val" / "Donut" / "Donut_0552.png"
if not img_path.exists():
    img_path = ROOT / "data" / "val" / "Donut" / "0552.png"

img = load_image(img_path)
state_dict = torch.load(ROOT / "wafer_model.pth", map_location="cpu", weights_only=True)

model_pt = WaferCNN()
model_pt.load_state_dict(state_dict)
model_pt.eval()

with torch.no_grad():
    x = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    c1_pt = model_pt.pool(model_pt.relu(model_pt.conv1(x))).numpy()[0]
    c2_pt = model_pt.pool(model_pt.relu(model_pt.conv2(torch.tensor(c1_pt).unsqueeze(0)))).numpy()[0]
    fc1_pt = model_pt.relu(model_pt.fc1(torch.tensor(c2_pt).view(1, -1))).numpy()[0]
    fc2_pt = model_pt.fc2(torch.tensor(fc1_pt).view(1, -1)).numpy()[0]

model_hw = FPGAInference()
# Run step by step
import shutil
spec4 = importlib.util.spec_from_file_location("simulate", str(ROOT / "software" / "4_simulate.py"))
sim = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(sim)
from config import N, IMAGE_SIZE

cols1 = sim_infer.im2col(img)
a1_scale = 127.0 / (np.max(np.abs(cols1)) + 1e-9)
sim_infer.write_activation_mem(cols1)
shutil.copy(ROOT / "weights" / "conv1_weight_int8.mem", ROOT / "conv1_weight_int8.mem")
c1_int32 = sim.run_accelerator(ROOT / "weights" / "conv1_weight_int8.npy", cols1.shape[0])
c1 = c1_int32 / (model_hw.w1_scale * a1_scale)
c1 = c1.reshape(N, IMAGE_SIZE, IMAGE_SIZE) + model_hw.b1[:, None, None]
c1_hw = model_hw.maxpool(model_hw.relu(c1))

print(f"C1 Match: {np.allclose(c1_hw, c1_pt, atol=1.0)}")
print(f"C1 HW mean: {np.mean(c1_hw):.4f}, PT mean: {np.mean(c1_pt):.4f}")

cols2 = sim_infer.im2col(c1_hw)
a2_scale = 127.0 / (np.max(np.abs(cols2)) + 1e-9)
sim_infer.write_activation_mem(cols2)
shutil.copy(ROOT / "weights" / "conv2_weight_int8.mem", ROOT / "conv1_weight_int8.mem")
c2_int32 = sim.run_accelerator(ROOT / "weights" / "conv2_weight_int8.npy", cols2.shape[0])
c2 = c2_int32 / (model_hw.w2_scale * a2_scale)
c2 = c2.reshape(N*2, IMAGE_SIZE//2, IMAGE_SIZE//2) + model_hw.b2[:, None, None]
c2_hw = model_hw.maxpool(model_hw.relu(c2))

print(f"C2 Match: {np.allclose(c2_hw, c2_pt, atol=1.0)}")
print(f"C2 HW mean: {np.mean(c2_hw):.4f}, PT mean: {np.mean(c2_pt):.4f}")

flat = c2_hw.flatten()
fc1_hw = model_hw.relu(flat @ model_hw.w3.T + model_hw.b3)
fc2_hw = fc1_hw @ model_hw.w4.T + model_hw.b4

print(f"FC1 Match: {np.allclose(fc1_hw, fc1_pt, atol=1.0)}")
print(f"FC2 Match: {np.allclose(fc2_hw, fc2_pt, atol=1.0)}")
