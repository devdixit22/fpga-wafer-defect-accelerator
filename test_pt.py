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

spec2 = importlib.util.spec_from_file_location("train", str(ROOT / "software" / "2_train.py"))
sim_train = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(sim_train)
WaferCNN = sim_train.WaferCNN

from config import CLASS_NAMES

img_path = ROOT / "data" / "val" / "Donut" / "Donut_0552.png"
if not img_path.exists():
    img_path = ROOT / "data" / "val" / "Donut" / "0552.png"

img = load_image(img_path)
state_dict = torch.load(ROOT / "wafer_model.pth", map_location="cpu", weights_only=True)

model = WaferCNN()
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    x = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    out = model(x)
    pred_idx = out.argmax(1).item()
    print(f"PyTorch prediction for {img_path.name}: {CLASS_NAMES[pred_idx]}")
    
    # print top 3
    probs = torch.softmax(out, dim=1)[0].numpy()
    top3 = np.argsort(probs)[-3:][::-1]
    for i in top3:
        print(f"  {CLASS_NAMES[i]:10s}: {probs[i]*100:.1f}%")
