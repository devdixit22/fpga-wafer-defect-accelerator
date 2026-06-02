"""
Step 3: Export and quantize model weights for the FPGA.

Reads wafer_model.pth.
Extracts weights and biases, quantizes to INT8, and saves them as:
    weights/*.npy (for software testing)
    weights/*.mem (hex format for Verilog $readmemh)

Usage:
    python software/3_export_weights.py
"""
import torch
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import N

WEIGHTS_DIR = ROOT / "weights"


def quantize_int8(x):
    max_val = np.max(np.abs(x))
    if max_val == 0:
        return x.astype(np.int8)
    scale = 127.0 / max_val
    return np.clip(np.round(x * scale), -128, 127).astype(np.int8)


def write_mem_file(data, filepath, flatten=True):
    """Write data as hex strings (one byte per line) for Verilog."""
    if flatten:
        data = data.flatten()
    with open(filepath, "w") as f:
        for val in data:
            f.write(f"{int(val) & 0xFF:02x}\n")


def format_conv_weights_for_fpga(w):
    """Pad and interleave conv weights for the systolic array memory."""
    # w is (out_channels, in_channels, kH, kW)
    # We flatten spatial dims: (out_c, in_f)
    w = w.reshape(w.shape[0], -1)
    out_c, in_f = w.shape

    # Pad to multiples of N
    pad_in = (N - (in_f % N)) % N
    if pad_in > 0:
        w = np.pad(w, ((0, 0), (0, pad_in)), mode="constant")
    
    pad_out = (N - (out_c % N)) % N
    if pad_out > 0:
        w = np.pad(w, ((0, pad_out), (0, 0)), mode="constant")

    out_blocks = w.shape[0] // N
    in_blocks = w.shape[1] // N

    # Format into (out_blocks, in_blocks, N, N)
    w_mem = np.zeros((out_blocks, in_blocks, N, N), dtype=np.int8)
    for ob in range(out_blocks):
        for ib in range(in_blocks):
            w_mem[ob, ib] = w[ob*N:(ob+1)*N, ib*N:(ib+1)*N]

    # For .mem file, each word is N*N weights (one entire block).
    # We write it out flattened, but reversed within each block for Verilog concatenation.
    hex_lines = []
    for ob in range(out_blocks):
        for ib in range(in_blocks):
            # The systolic array expects rows=in_features and cols=out_channels.
            # w_mem[ob, ib] has rows=out_channels and cols=in_features.
            # So we must transpose the (N, N) block before flattening it for hardware.
            block_flat = w_mem[ob, ib].T.flatten()
            hex_str = "".join([f"{int(val) & 0xFF:02x}" for val in reversed(block_flat)])
            hex_lines.append(hex_str)
    
    return w_mem, hex_lines


def main():
    model_path = ROOT / "wafer_model.pth"
    assert model_path.exists(), "Run 2_train.py first!"
    WEIGHTS_DIR.mkdir(exist_ok=True)

    print(f"Loading {model_path.name}...")
    state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)

    for name, param in state_dict.items():
        base_name = name.replace(".", "_")
        w_float = param.numpy()
        w_int8 = quantize_int8(w_float)

        # 1. Save standard .npy for software inference
        npy_path = WEIGHTS_DIR / f"{base_name}_int8.npy"
        np.save(str(npy_path), w_int8)

        # 2. Save .mem for hardware (only for conv weights/biases right now)
        if "conv" in name and "weight" in name:
            _, hex_lines = format_conv_weights_for_fpga(w_int8)
            mem_path = WEIGHTS_DIR / f"{base_name}_int8.mem"
            with open(mem_path, "w") as f:
                f.write("\n".join(hex_lines) + "\n")
            print(f"  Exported {name:15s} → .npy and .mem (FPGA formatted)")
        else:
            print(f"  Exported {name:15s} → .npy")

    print(f"\nAll weights quantized and saved to {WEIGHTS_DIR}/")


if __name__ == "__main__":
    main()
