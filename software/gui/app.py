import sys
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QVBoxLayout
)

from PyQt5.QtGui import QPixmap, QImage

from software.preprocessing.im2col import preprocess_image
from software.model.inference import FPGAInference


class WaferGUI(QWidget):

    def __init__(self):
        super().__init__()

        # Load FPGA inference model
        self.model = FPGAInference()

        self.setWindowTitle("FPGA Wafer Defect Detector")
        self.setGeometry(200, 200, 400, 400)

        self.layout = QVBoxLayout()

        self.label = QLabel("Upload a wafer image")
        self.layout.addWidget(self.label)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.button = QPushButton("Upload Image")
        self.button.clicked.connect(self.load_image)
        self.layout.addWidget(self.button)

        self.result = QLabel("Prediction:")
        self.layout.addWidget(self.result)

        self.setLayout(self.layout)

    def load_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if file_path:

            img = cv2.imread(file_path)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            h, w, ch = rgb.shape
            bytes_per_line = ch * w

            qt_img = QImage(
                rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(qt_img)
            self.image_label.setPixmap(pixmap)

            processed = preprocess_image(file_path)

            prediction = self.run_inference(processed)

            self.result.setText(f"Prediction: {prediction}")

    def run_inference(self, img):

        prediction = self.model.forward(img)

        return prediction


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = WaferGUI()
    window.show()

    sys.exit(app.exec_())
