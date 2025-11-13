from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import os
import json
import logging
import uuid
import traceback
from functools import wraps
from ocr import ocr_detection, ocr_filter_blocks, preload_models

# Configure logging - log file in project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(PROJECT_ROOT, 'ocr.log')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize rate limiter
# Custom key function: use API key from header if available, otherwise fall back to IP address
def get_rate_limit_key():
    """Get rate limit key from API key header, fallback to IP address"""
    api_key = request.headers.get('X-API-Key')
    if api_key:
        return f"api_key:{api_key}"
    return get_remote_address()

limiter = Limiter(
    app=app,
    key_func=get_rate_limit_key,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Custom error handler for rate limit exceeded
@app.errorhandler(429)
def ratelimit_handler(e):
    """Custom error handler for rate limit exceeded"""
    api_key = request.headers.get('X-API-Key', 'unknown')
    logger.warning(f"Rate limit exceeded for API key: {api_key[:10]}...")
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': str(e.description) if hasattr(e, 'description') else 'Too many requests',
        'retry_after': getattr(e, 'retry_after', None)
    }), 429

# Load API keys from JSON file
API_KEYS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_keys.json')
VALID_API_KEYS = set()

def load_api_keys():
    """Load valid API keys from JSON file"""
    global VALID_API_KEYS
    try:
        if os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                VALID_API_KEYS = set(data.get('valid_api_keys', []))
                logger.info(f"Loaded {len(VALID_API_KEYS)} API keys from {API_KEYS_FILE}")
        else:
            logger.warning(f"API keys file not found: {API_KEYS_FILE}")
            VALID_API_KEYS = set()
    except Exception as e:
        logger.error(f"Error loading API keys: {str(e)}")
        VALID_API_KEYS = set()

# Load API keys at startup
load_api_keys()

# Configure image upload directory
OCR_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(OCR_DIR, 'images')
# Ensure images directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def require_api_key(f):
    """Decorator to require valid API key for protected endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning("API key missing from request headers")
            return jsonify({'error': 'API key required. Please provide X-API-Key header.'}), 401
        
        if api_key not in VALID_API_KEYS:
            logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
            return jsonify({'error': 'Invalid API key'}), 401
        
        logger.debug("API key validated successfully")
        return f(*args, **kwargs)
    
    return decorated_function


@app.route('/health', methods=['GET'])
@limiter.limit("100 per minute")
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/ocr/process', methods=['POST'])
@require_api_key
@limiter.limit("60 per minute")
def process_image():
    """
    Complete OCR processing endpoint
    Performs both detection and filtering in one request
    Expects multipart/form-data with 'photo' file field
    Optional 'lang' parameter (default: 'japan') - can be sent as form field or JSON
    Optional 'skip_resize' (bool) - can be sent as form field or JSON
    Optional 'return_resized_coords' (bool) - can be sent as form field or JSON
    """
    saved_image_path = None
    try:
        # Handle file upload
        try:
            if 'photo' not in request.files:
                return jsonify({'error': 'No photo file provided'}), 400
        except Exception as files_error:
            logger.error(f"Error accessing request.files: {str(files_error)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': 'Invalid request format'}), 400
        
        file = request.files['photo']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg, gif, webp'}), 400
        
        # Generate unique photo_id
        photo_id = str(uuid.uuid4())
        
        # Save file with photo_id as filename
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        saved_filename = f"{photo_id}.{file_extension}"
        saved_image_path = os.path.join(IMAGES_DIR, saved_filename)
        
        # Save the file
        file.save(saved_image_path)
        logger.info(f"Saved uploaded file to: {saved_image_path}")
        
        # Get optional parameters from form data or JSON
        # Try to get from form data first, then from JSON if available
        if request.form:
            lang = request.form.get('lang', 'japan')
            skip_resize = bool(request.form.get('skip_resize', 'false').lower() == 'true')
            return_resized_coords = bool(request.form.get('return_resized_coords', 'false').lower() == 'true')
        else:
            # Fallback to JSON if form data is not available
            data = request.get_json(silent=True) or {}
            lang = data.get('lang', 'japan')
            skip_resize = bool(data.get('skip_resize', False))
            return_resized_coords = bool(data.get('return_resized_coords', False))
        
        logger.info(f"Received image processing request")
        logger.info(f"Saved image path: {saved_image_path}")
        logger.info(f"Language: {lang}")
        logger.info(f"Skip resize: {skip_resize}")
        logger.info(f"Return resized coords: {return_resized_coords}")
        
        # Check if file exists and is readable
        if not os.path.exists(saved_image_path):
            logger.error(f"Image file not found at path: {saved_image_path}")
            return jsonify({'error': f'Image file not found: {saved_image_path}'}), 404
        
        file_size = os.path.getsize(saved_image_path)
        logger.info(f"File exists, size: {file_size} bytes")
        
        if not os.access(saved_image_path, os.R_OK):
            logger.error(f"File exists but is not readable: {saved_image_path}")
            return jsonify({'error': f'Image file is not readable: {saved_image_path}'}), 403
        
        logger.info(f"File is readable, starting OCR detection...")
        
        # Perform OCR detection
        ocr_result = ocr_detection(saved_image_path, lang=lang, skip_resize=skip_resize)
        logger.info(f"OCR detection completed")
        logger.debug(f"Raw OCR result length: {len(ocr_result) if ocr_result else 0}")
        
        # Filter text blocks
        text_blocks, image_dimensions = ocr_filter_blocks(ocr_result, scale_back_to_original=not return_resized_coords)
        logger.info(f"Filtered {len(text_blocks)} text blocks")
        logger.debug(f"Image dimensions: {image_dimensions.get('image_width')}x{image_dimensions.get('image_height')}")
        logger.debug(f"Scale factor: {image_dimensions.get('scale_factor', 1.0):.3f}")
        
        response_data = {
            'success': True,
            'message': 'OCR processing completed',
            'data': {
                'text_blocks': text_blocks,
                'total_blocks': len(text_blocks),
                'language': lang,
                'image_path': saved_image_path,
                'image_width': image_dimensions.get('image_width'),
                'image_height': image_dimensions.get('image_height'),
                'scale_factor': image_dimensions.get('scale_factor', 1.0),
                'resized_width': image_dimensions.get('resized_width'),
                'resized_height': image_dimensions.get('resized_height')
            }
        }
        
        logger.info(f"Successfully processed image, returning response")
        logger.info(f"Response summary: {len(text_blocks)} blocks, dims: {image_dimensions.get('image_width')}x{image_dimensions.get('image_height')}, scale: {image_dimensions.get('scale_factor', 1.0):.3f}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.error(f"Exception occurred: {str(e)}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({'error': f'OCR processing failed: {str(e)}'}), 500
    
    finally:
        # Clean up temporary file after processing
        if saved_image_path and os.path.exists(saved_image_path):
            try:
                os.remove(saved_image_path)
                logger.info(f"Cleaned up temporary file: {saved_image_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary file {saved_image_path}: {str(cleanup_error)}")


if __name__ == '__main__':
    # Pre-load OCR models for faster initial requests
    # Only load models once (avoid duplicate loading when Flask reloader restarts)
    # Currently optimized for Japanese menus - other languages (ch, en) load on-demand
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':
        logger.info("=== Flask initial startup, skipping model preload (will load after reloader starts) ===")
    else:
        logger.info("=== Pre-loading Japanese OCR model at startup ===")
        preload_models(['japan'])  # Optimized for Menu Translator - loads Japanese only
        logger.info("=== Model pre-loading completed ===")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
