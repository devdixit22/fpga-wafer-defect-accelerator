import numpy as np
import glob

files = glob.glob("*_int8.npy")

for file in files:
    data = np.load(file).flatten()

    mem_file = file.replace(".npy", ".mem")

    with open(mem_file, "w") as f:
        for val in data:
            f.write(f"{int(val)}\n")

    print("Created:", mem_file)
