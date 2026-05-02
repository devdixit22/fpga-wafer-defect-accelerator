import subprocess
import numpy as np


def run_accelerator():

    # run verilog simulation
    subprocess.run([
        "iverilog",
        "-o", "accel.out",
        "hardware/rtl/pe.v",
        "hardware/rtl/systolic_array.v",
        "hardware/rtl/activation_buffer.v",
        "hardware/rtl/weight_buffer.v",
        "hardware/rtl/output_buffer.v",
        "hardware/tb/pe_tb.v"
    ])

    subprocess.run(["vvp", "accel.out"])

    # read accelerator output
    results = np.loadtxt("output.mem")

    return results
