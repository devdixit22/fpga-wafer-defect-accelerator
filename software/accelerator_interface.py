import numpy as np
import subprocess
import os


def run_accelerator(activation):

    # Save activation matrix for FPGA simulation
    np.savetxt(
        "activation.mem",
        activation.flatten(),
        fmt="%d"
    )

    # Compile Verilog accelerator
    subprocess.run(
        "iverilog -o accel_sim hardware/tb/accelerator_tb.v hardware/rtl/*.v",
        shell=True
    )

    # Run simulation
    subprocess.run(
        "vvp accel_sim",
        shell=True
    )

    # Read accelerator output
    if not os.path.exists("output.mem"):
        raise RuntimeError("Accelerator did not generate output.mem")

    output = np.loadtxt("output.mem")

    return output
