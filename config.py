"""
Central configuration for the FPGA Wafer Defect Accelerator.
All N, IMAGE_SIZE, and NUM_CLASSES values should be imported from here
to keep the hardware and software in sync.

N=8  → 64 PEs,  suitable for simulation (iverilog)
N=32 → 1024 PEs, production FPGA synthesis
"""

N = 8              # Systolic array width (rows & cols of PEs)
IMAGE_SIZE = 32    # Input image resolution (grayscale, square)
NUM_CLASSES = 9    # Defect classes: Center, Donut, Edge-Loc, Edge-Ring,
                   #   Loc, Random, Scratch, Near-full, None

CLASS_NAMES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Random", "Scratch", "Near-full", "None"
]
