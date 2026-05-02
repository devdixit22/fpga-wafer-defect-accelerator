import torch
import numpy as np

MODEL_PATH = "wafer_model.pth"

model_weights = torch.load(MODEL_PATH, map_location="cpu")

for name, param in model_weights.items():

    weights = param.numpy()

    filename = name.replace(".", "_") + ".npy"

    np.save(filename, weights)

    print("Saved:", filename)
