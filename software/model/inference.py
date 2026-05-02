import numpy as np

from software.preprocessing.im2col import im2col, export_activation_mem
from software.model.run_accelerator import run_accelerator


class FPGAInference:

    def __init__(self):

        # load quantized weights
        self.conv1_w = np.load("conv1_weight_int8.npy")
        self.conv1_b = np.load("conv1_bias_int8.npy")

        self.conv2_w = np.load("conv2_weight_int8.npy")
        self.conv2_b = np.load("conv2_bias_int8.npy")

        self.fc1_w = np.load("fc1_weight_int8.npy")
        self.fc1_b = np.load("fc1_bias_int8.npy")

        self.fc2_w = np.load("fc2_weight_int8.npy")
        self.fc2_b = np.load("fc2_bias_int8.npy")


    def relu(self, x):
        return np.maximum(0, x)


    def forward(self, img):

        # Convert image to im2col matrix
        cols = export_activation_mem(im2col(img))

        # Write activations for FPGA
        np.savetxt("activation.mem", cols.flatten(), fmt="%d")

        # Run accelerator (Verilog simulation)
        conv1 = run_accelerator()

        conv1 = conv1 + self.conv1_b
        conv1 = self.relu(conv1)

        # Run accelerator again for second convolution
        conv2 = run_accelerator()

        conv2 = conv2 + self.conv2_b
        conv2 = self.relu(conv2)

        # Fully connected layer 1 (software)
        fc1 = conv2 @ self.fc1_w.T + self.fc1_b
        fc1 = self.relu(fc1)

        # Fully connected layer 2 (software)
        fc2 = fc1 @ self.fc2_w.T + self.fc2_b

        pred = np.argmax(fc2)

        classes = [
            "Center",
            "Donut",
            "Edge-Loc",
            "Edge-Ring",
            "Loc",
            "Random",
            "Scratch",
            "Near-full",
            "None"
        ]

        return classes[pred]
