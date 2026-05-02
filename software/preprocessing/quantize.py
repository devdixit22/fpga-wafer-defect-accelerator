import numpy as np
import torch

MODEL_PATH = "wafer_model.pth"

state_dict = torch.load(MODEL_PATH, map_location="cpu")

def quantize_to_int8(x):

    max_val = np.max(np.abs(x))

    if max_val == 0:
        return x.astype(np.int8), 1.0

    scale = 127.0 / max_val
    q = (x * scale).astype(np.int8)

    return q, scale


for name, param in state_dict.items():

    weights = param.numpy()

    q_weights, scale = quantize_to_int8(weights)

    filename = name.replace(".", "_")

    np.save(f"{filename}_int8.npy", q_weights)

    print(f"Saved {filename}_int8.npy   scale={scale}")
