"""
End-to-end integration test for the FPGA Wafer Defect Accelerator.
Run from repo root:  python test_pipeline.py
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(label, cmd):
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(f"  [FAIL] {r.stderr[-500:]}")
        return False
    print("  [PASS]")
    return True

results = []

results.append(run("Step 1: Data Prep", [sys.executable, "software/1_prepare_data.py"]))
results.append(run("Step 2: Train Model", [sys.executable, "software/2_train.py"]))
results.append(run("Step 3: Export Weights", [sys.executable, "software/3_export_weights.py"]))

# Verify RTL
rtl = lambda f: str(ROOT / "hardware" / "rtl" / f)
tb  = lambda f: str(ROOT / "hardware" / "tb"  / f)

def sim_tb(name, files):
    out = ROOT / f"{name}.out"
    subprocess.run(["iverilog", "-o", str(out)] + files, check=True)
    subprocess.run(["vvp", str(out)], check=True)

try:
    print("\n  Step 4: Running basic RTL testbenches...")
    sim_tb("relu_tb", [rtl("relu.v"), tb("relu_tb.v")])
    sim_tb("pe_tb", [rtl("pe.v"), tb("pe_tb.v")])
    sim_tb("maxpool_tb", [rtl("maxpool_parallel.v"), tb("maxpool_parallel_tb.v")])
    print("  [PASS]")
    results.append(True)
except Exception as e:
    print(f"  [FAIL] {e}")
    results.append(False)

# Run Inference
try:
    print("\n  Step 5: Full end-to-end inference")
    import cv2
    import numpy as np
    from config import IMAGE_SIZE
    test_img = ROOT / "test_image.png"
    cv2.imwrite(str(test_img), np.random.randint(0, 255, (IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8))
    results.append(run("Inference Run", [sys.executable, "software/5_infer.py", str(test_img)]))
except Exception as e:
    print(f"  [FAIL] {e}")
    results.append(False)

print(f"\nResult: {sum(results)}/{len(results)} steps passed")
