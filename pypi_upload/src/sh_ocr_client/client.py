import requests
import os

def process_ocr(image_path, language, api_key):
    """
    Process an image using the OCR service.
    
    Args:
        image_path (str): Path to the image file to process
        language (str): Language code for OCR (e.g., 'japan', 'ch', 'en')
        api_key (str): API key for authentication (required)
    
    Returns:
        dict: OCR response containing text blocks and metadata
    
    Raises:
        requests.RequestException: If the request fails
        requests.HTTPError: If the API returns an error status (e.g., 401 for invalid API key)
        FileNotFoundError: If the image file does not exist
    """
    OCR_BASE_URL = 'http://ocrtest.9top.org/ocr/'
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    headers = {
        'X-API-Key': api_key
    }
    
    # Determine MIME type based on file extension
    file_ext = os.path.splitext(image_path)[1].lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    mime_type = mime_types.get(file_ext, 'image/jpeg')
    
    # Open the file and send it as multipart/form-data
    with open(image_path, 'rb') as image_file:
        files = {
            'photo': (os.path.basename(image_path), image_file, mime_type)
        }
        
        data = {
            'lang': language,
            'return_resized_coords': 'true',
            'skip_resize': 'true'
        }
        
        ocr_response = requests.post(
            OCR_BASE_URL + 'process',  # http://localhost:5001/ocr/process
            files=files,
            data=data,
            headers=headers,
            timeout=120
        )
    
    ocr_response.raise_for_status()
    return ocr_response.json()

