"""
Step 4: Run the Verilog Accelerator Simulation.

Provides a helper function to compile and run the Verilog testbench 
for a single convolutional layer, acting as a bridge between Python and Icarus Verilog.

Usage:
    (Imported by 5_infer.py to execute layers on the simulated FPGA)
"""
import subprocess
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import N


def run_accelerator(weight_npy_path, num_patches):
    """
    Compile and run the Verilog simulation for a conv layer.
    Reads weight metadata to configure the testbench parameters.
    Returns: The simulated output feature map as a numpy array.
    """
    rtl_dir = ROOT / "hardware" / "rtl"
    tb_dir = ROOT / "hardware" / "tb"

    # 1. Determine dimensions from the weight file
    w = np.load(weight_npy_path)
    w = w.reshape(w.shape[0], -1)  # Flatten spatial dims
    out_c, in_f = w.shape

    padded_in_f = in_f + ((N - (in_f % N)) % N)
    padded_out_c = out_c + ((N - (out_c % N)) % N)
    
    in_blocks = padded_in_f // N
    out_blocks = padded_out_c // N

    # 2. Compile RTL with parameters
    rtl_files = [
        str(rtl_dir / "pe.v"),
        str(rtl_dir / "relu.v"),
        str(rtl_dir / "bias_add.v"),
        str(rtl_dir / "maxpool_parallel.v"),
        str(rtl_dir / "array_controller.v"),
        str(rtl_dir / "activation_buffer.v"),
        str(rtl_dir / "weight_buffer.v"),
        str(rtl_dir / "output_buffer.v"),
        str(rtl_dir / "systolic_array.v"),
        str(rtl_dir / "accelerator_top.v"),
        str(rtl_dir / "system_top.v"),
        str(tb_dir  / "system_tb.v"),
    ]

    out_bin = ROOT / "accel.out"
    compile_cmd = [
        "iverilog", "-o", str(out_bin),
        f"-Psystem_tb.N={N}",
        f"-Psystem_tb.IN_BLOCKS={in_blocks}",
        f"-Psystem_tb.OUT_BLOCKS={out_blocks}",
        f"-Psystem_tb.NUM_PATCHES={num_patches}",
    ] + rtl_files

    c_res = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=str(ROOT))
    if c_res.returncode != 0:
        raise RuntimeError(f"iverilog compile failed:\n{c_res.stderr}")

    # 3. Simulate
    # Note: testbench expects 'activation.mem' and the specific weight .mem in ROOT
    # 5_infer.py handles copying them there.
    s_res = subprocess.run(["vvp", str(out_bin)], capture_output=True, text=True, cwd=str(ROOT))
    if s_res.returncode != 0:
        raise RuntimeError(f"vvp sim failed:\n{s_res.stderr}")

    # 4. Read output.mem
    out_mem = ROOT / "output.mem"
    if not out_mem.exists():
        raise RuntimeError("Simulation did not produce output.mem")

    # Output is (out_blocks * num_patches, N) words, flattened in text
    data = np.loadtxt(str(out_mem), dtype=np.int32)
    data = data.reshape(out_blocks, num_patches, N)
    data = data.transpose(0, 2, 1).reshape(padded_out_c, num_patches)
    
    # Trim padding and return
    return data[:out_c, :]
