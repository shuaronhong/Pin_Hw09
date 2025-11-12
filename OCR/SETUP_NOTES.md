# OCR Setup Notes

## Current Configuration
- **PaddlePaddle**: 2.6.1 (CPU version)
- **PaddleOCR**: 2.7.0
- **NumPy**: 1.26.4 (1.x series for compatibility)
- **Mode**: CPU (cuDNN not installed)

## What Was Fixed
1. Removed `use_gpu=True` parameter (deprecated in PaddlePaddle 3.0+, not needed in 2.6.1)
2. Added `paddle.set_device()` at module level for future GPU support
3. Excluded PyMuPDF and pdf2docx (they take 10+ minutes to build from source and are only needed for PDF files)
4. Fixed numpy version to 1.x for OpenCV compatibility
5. Simplified installation script to avoid build issues

## Installation
Simply run:
```bash
./run_ocr.sh
```

The script will:
1. Create/activate virtual environment
2. Install all dependencies from requirements.txt
3. Install PaddleOCR (skipping optional PDF dependencies)
4. Start the OCR service on port 5001

## GPU Support (Optional)
To enable GPU acceleration in the future:
1. Install cuDNN library on your system
2. Replace `paddlepaddle==2.6.1` with `paddlepaddle-gpu==2.6.1` in requirements.txt
3. The code already has GPU detection/fallback logic in ocr.py

## Performance
- **CPU Mode**: Works but slower (current configuration)
- **GPU Mode**: Would be 10-100x faster with cuDNN installed

## What Works
✅ Image-based menu OCR (Japanese, Chinese, English)
✅ Flask web service on port 5001
✅ Translation backend integration
✅ No build dependencies required

## What's Excluded
❌ PDF file processing (requires PyMuPDF)
Note: PDFs weren't needed for the menu translator use case (works with images)

