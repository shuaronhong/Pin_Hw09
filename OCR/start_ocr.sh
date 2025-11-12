#!/bin/bash
# Wrapper script to ensure proper virtual environment activation

cd "$(dirname "$0")"
source venv/bin/activate
export PYTHONPATH="$VIRTUAL_ENV/lib/python3.13/site-packages:$PYTHONPATH"

# Set thread limits to fix PaddlePaddle CPU backend concurrency issues
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec ./venv/bin/python ocr_app.py
