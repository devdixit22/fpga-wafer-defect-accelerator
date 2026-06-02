# FPGA Wafer Defect Accelerator

An end-to-end pipeline for detecting defects on silicon semiconductor wafers. This project trains a Convolutional Neural Network (CNN) in PyTorch, quantizes the weights, and executes the mathematical inference directly on a **Custom Hardware Verilog Systolic Array** simulator. It includes a modern web-based dashboard to visualize the hardware inference results in real-time.

![Web Dashboard](data/custom_processed.png) <!-- Update with an actual dashboard screenshot if available -->

## Features
- **PyTorch CNN Training**: A robust Python pipeline to train models on the WM-811K wafer defect dataset.
- **Hardware Co-Simulation**: Python cleanly hooks into the Icarus Verilog (`iverilog`) simulator to run bit-accurate INT8 hardware math natively.
- **Custom Systolic Array**: RTL implementations of processing elements (PEs), an 8x8 Systolic Array, and a memory controller to execute 2D Convolutions in hardware.
- **Real-Time Web Dashboard**: A Flask-based interactive UI to upload wafer images, run the FPGA pipeline in the background, and display inference times, heatmaps, and predictions.
- **OpenCV Image Converter**: Preprocesses physical, real-world photographs of silicon wafers into the simplified binary formats the neural network was trained on.

## Project Structure

### 1. `hardware/`
Contains the Verilog RTL files for the hardware accelerator.
- `rtl/systolic_array.v`: The core matrix multiplication grid.
- `rtl/pe.v`: The individual processing elements for multiply-accumulate (MAC).
- `rtl/system_top.v`: The top-level wrapper handling memory buffering and sequential chunking for large feature maps.
- `tb/system_tb.v`: The Verilog testbench used to run the simulation.

### 2. `software/`
The automated Python pipeline for training and deployment.
- `1_prepare_data.py`: Standardizes and extracts the dataset.
- `2_train.py`: Trains the PyTorch CNN and saves `wafer_model.pth`.
- `3_export_weights.py`: Quantizes FP32 PyTorch weights into INT8, transposes them, and formats them into `.mem` files tailored for the Verilog physical memory layout.
- `4_simulate.py`: Acts as the bridge between Python and `iverilog`, dynamically recompiling and executing the Verilog testbench.
- `5_infer.py`: Orchestrates full end-to-end inference, running Convolution layers on the simulated hardware and Fully Connected layers in software.
- `convert_wafer.py`: OpenCV script to process real-world photographs into WM-811K style formats.
- `web_app/app.py`: The Flask Web Dashboard server.

## Installation & Requirements

Ensure you have Python 3.8+ and Icarus Verilog (`iverilog`) installed on your system.

```bash
# Clone the repository
git clone https://github.com/devdixit22/fpga-wafer-defect-accelerator.git
cd fpga-wafer-defect-accelerator

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install torch torchvision numpy opencv-python Flask pandas matplotlib
```

*(Note: Ubuntu users can install Icarus Verilog via `sudo apt install iverilog`)*

## How to Run the Web Dashboard

To launch the interactive dashboard and test the hardware-in-the-loop inference:

```bash
export PYTHONPATH=$(pwd)
source venv/bin/activate
python software/web_app/app.py
```
Then, open your browser and navigate to `http://localhost:5000`.

## Testing Real Photographs

If you have a physical picture of a silicon wafer and want to test it:

```bash
# 1. Convert the photo using the OpenCV preprocessor
python software/convert_wafer.py path/to/your_photo.jpg data/my_processed_wafer.png

# 2. Upload 'data/my_processed_wafer.png' to the Web Dashboard!
```

## Hardware Verification & Debugging Scripts
To verify that the hardware simulation is mathematically flawless and perfectly matches the floating-point PyTorch model, you can run the validation test suites:
- `test_fpga_layers.py`: Validates the exact INT8 quantizations and outputs layer-by-layer between PyTorch and Verilog.
- `test_fpga.py`: Runs a raw end-to-end prediction via the command line without launching the web server.

## Architecture & Hardware Logic
The accelerator utilizes a `N=8` (8x8) Systolic Array. Because typical image feature maps (e.g., 32x32) exceed the physical dimensions of the grid, the `system_top.v` state machine implements a chunking/tiling algorithm. It breaks large matrices into N-sized blocks, feeds them through the array, and accumulates the partial sums before passing the final results through software-driven ReLUs. 
