#!/bin/bash

echo "--------------------------------"
echo "FPGA Wafer Defect Accelerator"
echo "--------------------------------"

echo "Activating environment..."
source venv/bin/activate

echo "Generating activation memory..."

python3 - <<EOF
from software.preprocessing.im2col import preprocess_image, im2col, export_activation_mem

img = preprocess_image("test_image.png")
cols = im2col(img)
export_activation_mem(cols)

print("activation.mem generated")
EOF


echo "Compiling accelerator..."

iverilog -o accelerator.out \
hardware/rtl/pe.v \
hardware/rtl/systolic_array.v \
hardware/rtl/activation_buffer.v \
hardware/rtl/weight_buffer.v \
hardware/rtl/output_buffer.v \
hardware/tb/pe_tb.v


echo "Running accelerator..."

vvp accelerator.out


echo "Reading accelerator output..."

python3 - <<EOF
import numpy as np

data = np.loadtxt("output.mem")

print("Accelerator output:")
print(data[:10])
EOF


echo "Launching GUI..."

python3 -m software.gui.app
