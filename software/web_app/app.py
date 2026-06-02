"""
FPGA Wafer Defect Detector — Web Dashboard
A modern Flask-based interface replacing the PyQt5 GUI.
Run:  python software/gui/web_app.py
"""
import sys
import os
import json
import csv
import io
import base64
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flask import Flask, render_template, request, jsonify, send_from_directory
from config import N, IMAGE_SIZE, CLASS_NAMES
import importlib.util
spec = importlib.util.spec_from_file_location("infer", str(REPO_ROOT / "software" / "5_infer.py"))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)
FPGAInference = sim.FPGAInference
load_image = sim.load_image

app = Flask(__name__,
            template_folder=str(REPO_ROOT / "software" / "web_app" / "templates"),
            static_folder=str(REPO_ROOT / "software" / "web_app" / "static"))

UPLOAD_DIR = REPO_ROOT / "software" / "web_app" / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_FILE = RESULTS_DIR / "defect_log.csv"

# Lazy-load model (heavy init)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = FPGAInference()
    return _model


def _make_heatmap_b64(img_gray):
    """Generate a base64-encoded heatmap visualization of the grayscale image."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    im = ax.imshow(img_gray, cmap='inferno', interpolation='nearest')
    ax.set_title("Intensity Heatmap", color='white', fontsize=11, pad=8)
    ax.axis('off')
    fig.patch.set_facecolor('#0f0f23')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _make_histogram_b64(img_gray):
    """Generate a base64-encoded pixel intensity histogram."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.hist(img_gray.flatten(), bins=50, color='#00d4ff', alpha=0.85, edgecolor='#0a0a1a')
    ax.set_xlabel("Pixel Intensity", color='#aaa', fontsize=9)
    ax.set_ylabel("Count", color='#aaa', fontsize=9)
    ax.set_title("Pixel Distribution", color='white', fontsize=11, pad=8)
    ax.tick_params(colors='#666')
    ax.set_facecolor('#0f0f23')
    fig.patch.set_facecolor('#0f0f23')
    for spine in ax.spines.values():
        spine.set_color('#333')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def _log_result(filename, prediction, duration_ms):
    """Append result to CSV log."""
    exists = LOG_FILE.exists()
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Timestamp", "Image", "Prediction", "Inference_ms"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                     filename, prediction, f"{duration_ms:.1f}"])


def _read_log():
    """Read entire CSV log and return as list of dicts."""
    if not LOG_FILE.exists():
        return []
    rows = []
    with open(LOG_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', N=N, IMAGE_SIZE=IMAGE_SIZE)


@app.route('/api/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save upload
    save_path = UPLOAD_DIR / file.filename
    file.save(str(save_path))

    try:
        t0 = datetime.now()

        # Preprocess
        img_array = load_image(str(save_path))

        # Run FPGA inference
        model = get_model()
        prediction = model.forward(img_array)

        duration_ms = (datetime.now() - t0).total_seconds() * 1000

        # Generate visualizations
        heatmap_b64 = _make_heatmap_b64(img_array)
        histogram_b64 = _make_histogram_b64(img_array)

        # Read original image for display
        img_bgr = cv2.imread(str(save_path))
        if img_bgr is not None:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode('.png', img_bgr)
            original_b64 = base64.b64encode(buf).decode('utf-8')
        else:
            original_b64 = ""

        # Log
        _log_result(file.filename, prediction, duration_ms)

        return jsonify({
            "prediction": prediction,
            "duration_ms": round(duration_ms, 1),
            "original_image": original_b64,
            "heatmap": heatmap_b64,
            "histogram": histogram_b64,
            "image_shape": list(img_array.shape),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/history')
def history():
    return jsonify(_read_log())


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


if __name__ == '__main__':
    print("\n" + "="*55)
    print(f"  FPGA Wafer Defect Detector — Web Dashboard")
    print(f"  Systolic Array: {N}×{N} ({N*N} PEs)")
    print(f"  Image Size: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print("  Open http://localhost:5000 in your browser")
    print("="*55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
