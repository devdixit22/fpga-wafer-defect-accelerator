#!/bin/bash
set -e

echo "================================================="
echo " FPGA Wafer Defect Accelerator - Pipeline Runner"
echo "================================================="

cd "$(dirname "$0")"
source venv/bin/activate

echo ""
echo "[1/4] Preparing Dataset..."
python software/1_prepare_data.py

echo ""
echo "[2/4] Training CNN..."
python software/2_train.py

echo ""
echo "[3/4] Exporting Weights to FPGA Format..."
python software/3_export_weights.py

echo ""
echo "[4/4] Launching Web Dashboard..."
export PYTHONPATH=$(pwd)
python software/web_app/app.py
