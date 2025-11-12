import os
import logging
import threading
import cv2
import re

# Set environment variables to fix PaddlePaddle CPU threading issues
# MUST be set BEFORE importing paddle/paddleocr
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Now import paddle and paddleocr
from paddleocr import PaddleOCR
from PIL import Image
import paddle

logger = logging.getLogger(__name__)

# Set device to GPU if available (PaddlePaddle 3.0+ way)
# This replaces the deprecated use_gpu parameter
try:
    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        paddle.set_device('gpu')
        logger.info("✓ GPU device set successfully")
    else:
        paddle.set_device('cpu')
        logger.info("GPU not available, using CPU")
except Exception as e:
    logger.warning(f"Could not set device, using default: {e}")

# Global cache for pre-loaded OCR models
_ocr_models = {}
_model_lock = threading.Lock()
# Global inference lock to prevent concurrent OCR calls (fixes "could not execute a primitive" error)
_inference_lock = threading.Lock()

def preload_models(languages=['japan', 'ch', 'en']):
    """
    Pre-load OCR models for specified languages at startup
    Args:
        languages: List of language codes to pre-load (default: Japanese, Chinese, English)
    """
    logger.info(f"Starting model pre-loading for languages: {languages}")
    
    for lang in languages:
        try:
            logger.info(f"Loading OCR model for language: {lang}...")
            # Disable ir_optim to prevent "could not execute a primitive" errors on CPU
            # This disables MKL-DNN optimizations which cause memory corruption issues
            ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=False,
                ir_optim=False,  # Disable inference optimizations to fix CPU stability issues
                enable_mkldnn=False,  # Explicitly disable MKL-DNN
                cpu_threads=1  # Force single-threaded execution
            )
            _ocr_models[lang] = ocr
            logger.info(f"✓ Successfully loaded OCR model for language: {lang}")
        except Exception as e:
            logger.error(f"✗ Error loading model for language {lang}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"Model pre-loading completed. Loaded {len(_ocr_models)} models: {list(_ocr_models.keys())}")
    return _ocr_models

def get_ocr_model(lang='japan'):
    """
    Get a pre-loaded OCR model or create a new one if not cached
    Thread-safe implementation to prevent duplicate model loading
    Args:
        lang: Language code for the OCR model
    Returns:
        PaddleOCR instance
    """
    # Fast path: model already loaded
    if lang in _ocr_models:
        return _ocr_models[lang]
    
    # Slow path: need to load model (use lock to prevent race condition)
    with _model_lock:
        # Double-check after acquiring lock (another thread might have loaded it)
        if lang in _ocr_models:
            return _ocr_models[lang]
        
        logger.info(f"Model not pre-loaded for language: {lang}, initializing new model...")
        try:
            # Note: GPU usage is controlled via paddle.set_device() at module level
            # The use_gpu parameter has been removed in PaddlePaddle 3.0+
            # Disable optimizations to prevent CPU stability issues
            ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=False,
                ir_optim=False,  # Disable inference optimizations
                enable_mkldnn=False,  # Disable MKL-DNN
                cpu_threads=1  # Single-threaded execution
            )
            _ocr_models[lang] = ocr
            logger.info(f"Successfully initialized and cached model for language: {lang}")
            return ocr
        except Exception as e:
            logger.error(f"Error initializing model for language {lang}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
def image_to_numpy(image_path: str):
    """
    Convert the image to a numpy array
    Args:
        image_path: The path to the image
    Returns:
        numpy_array: The numpy array of the image
    """
    img = cv2.imread(image_path)  # ✅ load as numpy array (BGR)
    if img is None:
        raise FileNotFoundError(f"❌ Cannot read image: {image_path}")

    return img
    
    
def ocr_detection(image_path: str, lang: str = 'japan', skip_resize: bool = False):
    """
    Detect the text in the image
    Args:
        image_path: The path to the image, could be a local path or a URL
        lang: The language of the original text
        skip_resize: Not used, kept for API compatibility
    Returns:
        ocr_result: The result of the OCR detection with image dimensions and scale factor
    """
    logger.info(f"Starting OCR detection for: {image_path}")
    try:
        with Image.open(image_path) as _img:
            original_width, original_height = _img.size
            resized_width, resized_height = original_width, original_height
    except Exception:
        original_width = original_height = None
        resized_width = resized_height = None
    resized_path = image_path
    scale_factor = 1.0
    
    try:
        # Get pre-loaded model or initialize if needed
        ocr = get_ocr_model(lang)
    except Exception as ocr_init_error:
        logger.error(f"Error getting OCR model: {str(ocr_init_error)}")
        raise

    try:
        img = image_to_numpy(resized_path)
        
        # Use inference lock to prevent concurrent OCR calls (fixes CPU backend issues)
        with _inference_lock:
            ocr_result = ocr.ocr(img)
        
        logger.info("OCR prediction completed")
        if not ocr_result or len(ocr_result) == 0:
            logger.warning("Empty OCR result")
    except Exception as predict_error:
        logger.error(f"Error during OCR prediction: {str(predict_error)}")
        raise
    
    # Append original and resized dimensions and scale factor for proper overlay
    ocr_result.append({
        'image_width': original_width,
        'image_height': original_height,
        'scale_factor': scale_factor,
        'resized_width': resized_width,
        'resized_height': resized_height
    })

    return ocr_result


def merge_dish_names_with_prices(items):
    """
    Merge dish names with nearby prices on the same line
    Args:
        items: List of text items with bbox coordinates
    Returns:
        merged_items: List with prices merged into dish names
    """
    if not items:
        return items
    
    # Separate dish names and potential prices
    dish_items = []
    price_items = []
    
    for item in items:
        text = item['text']
        # Check if it's a price-like pattern (contains numbers and price indicators)
        # Matches: £6.30, $5.99, 6.30, 10元, ¥58, 12 Yuan, etc.
        is_price = bool(re.search(r'[£$€¥]\s*\d+\.?\d*|\d+\.?\d*\s*[£$€¥元Yuanyuan]|\b\d+\.\d{2}\b', text, re.IGNORECASE))
        
        # Check if it contains CJK characters (likely dish name)
        has_cjk = any('\u4e00' <= char <= '\u9fff' or  # Chinese
                     '\u3040' <= char <= '\u309f' or  # Hiragana
                     '\u30a0' <= char <= '\u30ff' or  # Katakana
                     '\uac00' <= char <= '\ud7af'     # Korean
                     for char in text)
        
        if has_cjk and not is_price:
            dish_items.append(item)
        elif is_price:
            price_items.append(item)
        else:
            dish_items.append(item)  # Keep other text as-is
    
    # Merge prices with nearby dish names
    merged_items = []
    used_prices = set()
    
    for dish in dish_items:
        dish_bbox = dish['bbox']
        dish_y_center = (dish_bbox[1] + dish_bbox[3]) / 2
        dish_x_right = dish_bbox[2]
        
        # Find closest price on the same line (horizontally aligned)
        closest_price = None
        min_distance = float('inf')
        
        for idx, price in enumerate(price_items):
            if idx in used_prices:
                continue
                
            price_bbox = price['bbox']
            price_y_center = (price_bbox[1] + price_bbox[3]) / 2
            price_x_left = price_bbox[0]
            
            # Check if on same line (vertical alignment within threshold)
            y_diff = abs(dish_y_center - price_y_center)
            height_threshold = (dish_bbox[3] - dish_bbox[1]) * 0.8
            
            if y_diff < height_threshold:
                # Price should be to the right of dish name
                x_distance = price_x_left - dish_x_right
                
                # Check if price is nearby (not too far)
                if 0 <= x_distance < 300:  # Adjust threshold as needed
                    if x_distance < min_distance:
                        min_distance = x_distance
                        closest_price = (idx, price)
        
        # Merge if found
        if closest_price:
            price_idx, price_item = closest_price
            used_prices.add(price_idx)
            
            # Merge text BUT keep the original dish name bbox for positioning
            # This ensures the translation overlay appears over the Chinese text, not stretched to the price
            merged_text = f"{dish['text']} {price_item['text']}"
            
            # Use the original dish bbox (don't expand to include price position)
            # The translation will show both dish name + price, but positioned over the dish name
            merged_bbox = dish_bbox  # Keep original dish position
            
            merged_items.append({
                'id': dish['id'],
                'text': merged_text,
                'confidence': min(dish['confidence'], price_item['confidence']),
                'bbox': merged_bbox
            })
            logger.debug(f"Merged: '{merged_text}' (bbox from dish name only)")
        else:
            merged_items.append(dish)
    
    # Add any unmerged prices as standalone items (shouldn't happen often)
    for idx, price in enumerate(price_items):
        if idx not in used_prices:
            merged_items.append(price)
    
    return merged_items


def ocr_filter_blocks(result: list, scale_back_to_original: bool = True):
    """
    Get the essential information from the OCR result for translation
    Args:
        result: The result of the OCR detection
        scale_back_to_original: Not used, kept for API compatibility
    Returns:
        items: The essential information for translation with bboxes
        image_dimensions: Dictionary with image_width, image_height, and scale_factor
    """
    if not result or len(result) == 0:
        logger.error("Empty result provided")
        return [], {}
    
    # Extract image dimensions if present (should be the last element)
    image_dimensions = {}
    if len(result) > 1 and isinstance(result[-1], dict) and 'image_width' in result[-1]:
        image_dimensions = result[-1]
        # Remove the metadata dict from processing
        ocr_data = result[:-1]
    else:
        ocr_data = result
    
    # Handle PaddleOCR's standard output format
    # Format: [[[bbox], (text, confidence)], ...]
    data = ocr_data[0] if ocr_data else []
    
    if not data:
        logger.warning("No OCR detection results")
        return [], image_dimensions
    
    # Check if data is in the expected PaddleOCR format (list of detections)
    if isinstance(data, list):
        # Standard PaddleOCR format: each element is [bbox, (text, score)]
        items = []
        filtered_count = 0
        
        for i, detection in enumerate(data):
            if detection is None or len(detection) < 2:
                filtered_count += 1
                continue
                
            box = detection[0]
            text_info = detection[1]
            
            # text_info is a tuple: (text, confidence_score)
            if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                text = text_info[0]
                score = text_info[1]
            else:
                filtered_count += 1
                continue
            
            if score > 0.4 and text and text.strip():
                cleaned_text = text.strip()
                
                # Check if text contains CJK characters (Chinese, Japanese, Korean)
                has_cjk = any('\u4e00' <= char <= '\u9fff' or  # Chinese
                             '\u3040' <= char <= '\u309f' or  # Hiragana
                             '\u30a0' <= char <= '\u30ff' or  # Katakana
                             '\uac00' <= char <= '\ud7af'     # Korean
                             for char in cleaned_text)
                
                # Check if it's a price pattern
                # Matches: £6.30, $5.99, 6.30, 10元, ¥58, 12 Yuan, etc.
                is_price_like = bool(re.search(r'[£$€¥]\s*\d+\.?\d*|\d+\.?\d*\s*[£$€¥元Yuanyuan]|\b\d+\.\d{2}\b', cleaned_text, re.IGNORECASE))
                
                # Filter out menu item numbers (standalone single/double digit numbers)
                # But KEEP prices (we'll merge them with dish names later)
                if not has_cjk and not is_price_like:
                    # Remove common punctuation for checking
                    text_without_punct = cleaned_text.rstrip('.,;:!?()[]')
                    
                    # Skip if it's just a menu number (but not a price)
                    if text_without_punct.isdigit() and len(text_without_punct) <= 3:
                        filtered_count += 1
                        logger.debug(f"Filtered out menu number: '{cleaned_text}'")
                        continue
                    
                    # Filter out very short non-meaningful text
                    if len(cleaned_text) <= 2:
                        filtered_count += 1
                        logger.debug(f"Filtered out short text: '{cleaned_text}'")
                        continue
                
                # Convert PaddleOCR's 4-point polygon to a simple bounding box
                # PaddleOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                box_array = box.tolist() if hasattr(box, 'tolist') else box
                
                # Extract min/max coordinates to create a proper bounding rectangle
                if len(box_array) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box_array):
                    # 4-point format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    xs = [pt[0] for pt in box_array]
                    ys = [pt[1] for pt in box_array]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox = [x_min, y_min, x_max, y_max]
                else:
                    # Fallback to original format
                    bbox = box_array
                
                items.append({
                    'id': i+1,
                    'text': cleaned_text,
                    'confidence': round(float(score), 3),
                    'bbox': bbox
                })
            else:
                filtered_count += 1
        
        logger.info(f"Filtered {len(items)} items (excluded {filtered_count} low-quality items)")
        
        # Merge dish names with nearby prices
        merged_items = merge_dish_names_with_prices(items)
        logger.info(f"After merging: {len(merged_items)} items")
        
        return merged_items, image_dimensions
    
    # Fallback: Handle dict format (legacy or custom format)
    elif isinstance(data, dict):
        texts = data.get('rec_texts', [])
        scores = data.get('rec_scores', []) 
        boxes = data.get('rec_boxes', [])
        
        items = []
        filtered_count = 0
        for i, (text, score, box) in enumerate(zip(texts, scores, boxes)):
            if score > 0.4 and text.strip():
                cleaned_text = text.strip()
                
                # Check if text contains CJK characters (Chinese, Japanese, Korean)
                has_cjk = any('\u4e00' <= char <= '\u9fff' or  # Chinese
                             '\u3040' <= char <= '\u309f' or  # Hiragana
                             '\u30a0' <= char <= '\u30ff' or  # Katakana
                             '\uac00' <= char <= '\ud7af'     # Korean
                             for char in cleaned_text)
                
                # Check if it's a price pattern
                # Matches: £6.30, $5.99, 6.30, 10元, ¥58, 12 Yuan, etc.
                is_price_like = bool(re.search(r'[£$€¥]\s*\d+\.?\d*|\d+\.?\d*\s*[£$€¥元Yuanyuan]|\b\d+\.\d{2}\b', cleaned_text, re.IGNORECASE))
                
                # Filter out menu item numbers (standalone single/double digit numbers)
                # But KEEP prices (we'll merge them with dish names later)
                if not has_cjk and not is_price_like:
                    # Remove common punctuation for checking
                    text_without_punct = cleaned_text.rstrip('.,;:!?()[]')
                    
                    # Skip if it's just a menu number (but not a price)
                    if text_without_punct.isdigit() and len(text_without_punct) <= 3:
                        filtered_count += 1
                        logger.debug(f"Filtered out menu number: '{cleaned_text}'")
                        continue
                    
                    # Filter out very short non-meaningful text
                    if len(cleaned_text) <= 2:
                        filtered_count += 1
                        logger.debug(f"Filtered out short text: '{cleaned_text}'")
                        continue
                
                # Convert PaddleOCR's 4-point polygon to a simple bounding box
                box_array = box.tolist() if hasattr(box, 'tolist') else box
                
                # Extract min/max coordinates to create a proper bounding rectangle
                if len(box_array) == 4 and all(isinstance(pt, (list, tuple)) and len(pt) == 2 for pt in box_array):
                    # 4-point format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    xs = [pt[0] for pt in box_array]
                    ys = [pt[1] for pt in box_array]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox = [x_min, y_min, x_max, y_max]
                else:
                    # Fallback to original format
                    bbox = box_array
                
                items.append({
                    'id': i+1,
                    'text': cleaned_text,
                    'confidence': round(score, 3),
                    'bbox': bbox
                })
            else:
                filtered_count += 1
        
        logger.info(f"Filtered {len(items)} items (excluded {filtered_count} low-quality items)")
        
        # Merge dish names with nearby prices
        merged_items = merge_dish_names_with_prices(items)
        logger.info(f"After merging: {len(merged_items)} items")
        
        return merged_items, image_dimensions
    
    else:
        logger.error(f"Unexpected OCR result format, type: {type(data)}")
        return [], image_dimensions
