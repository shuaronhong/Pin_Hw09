#!/usr/bin/env python3
import sys
import os

print("Python executable:", sys.executable)
print("Python path:", sys.path)
print("Current working directory:", os.getcwd())
print("Environment PATH:", os.environ.get('PATH', ''))
print("PYTHONPATH:", os.environ.get('PYTHONPATH', ''))

try:
    import paddle
    print("✓ Paddle imported successfully")
except ImportError as e:
    print("✗ Failed to import paddle:", e)

try:
    import paddleocr
    print("✓ PaddleOCR imported successfully")
except ImportError as e:
    print("✗ Failed to import paddleocr:", e)
