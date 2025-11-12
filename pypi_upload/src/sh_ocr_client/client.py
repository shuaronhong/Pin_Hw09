import requests

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
    """
    OCR_BASE_URL = 'http://localhost:5001/ocr/'
    
    headers = {
        'X-API-Key': api_key
    }
    
    ocr_response = requests.post(
        OCR_BASE_URL + 'process',  # http://localhost:5001/ocr/process
        json={
            'image_path': image_path,
            'lang': language,
            'return_resized_coords': True,
            'skip_resize': True
        },
        headers=headers,
        timeout=120
    )
    
    ocr_response.raise_for_status()
    return ocr_response.json()

