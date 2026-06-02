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

from config import CLASS_NAMES

img_path = ROOT / "data" / "custom_processed.png"

img = load_image(img_path)
model = FPGAInference()
pred = model.forward(img)
print(f"FPGA Inference prediction: {pred}")
