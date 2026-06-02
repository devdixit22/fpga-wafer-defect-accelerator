import cv2
import numpy as np
import sys
from pathlib import Path

def process_wafer(input_path, output_path):
    print(f"Loading {input_path}...")
    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError("Could not read image")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Isolate the Wafer Circle
    # Threshold to find the circle (background is usually white/gray, wafer is darker, or vice-versa)
    # The uploaded image has a white background and a grey wafer with white defects.
    # Let's threshold to find the circle.
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No wafer found!")
        return
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create mask for the wafer
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # 2. Extract internal features
    # Inside the wafer, normal die is grey, defects are white.
    # We want to map: background -> 0, normal -> 128, defects -> 255
    wafer_only = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Calculate median intensity of the wafer for normal die
    mean_val = cv2.mean(wafer_only, mask=mask)[0]
    
    # Create output canvas
    out = np.zeros_like(gray)
    
    # Set normal die to 128
    out[mask == 255] = 128
    
    # Threshold defects. Anything significantly brighter or darker than the mean could be a defect.
    # In the uploaded image, defects are bright white scratches at the bottom.
    # Let's threshold bright spots inside the mask
    _, defects_bright = cv2.threshold(wafer_only, 180, 255, cv2.THRESH_BINARY)
    
    # Add defects to output (set to 255)
    out[defects_bright == 255] = 255
    
    # Optional: morphology to clean up noise
    kernel = np.ones((3,3), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    
    # Resize to something reasonable or just save as is
    cv2.imwrite(str(output_path), out)
    print(f"Saved processed image to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_wafer.py <in> <out>")
        sys.exit(1)
    process_wafer(sys.argv[1], sys.argv[2])
