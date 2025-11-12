# sh-ocr-client

A Python client library for accessing OCR (Optical Character Recognition) service. This library allows you to extract text from images by providing the image path and language code.

## Features

- Simple API to process images through OCR service
- Support for multiple languages (e.g., 'japan', 'ch', 'en')
- Returns structured text blocks with coordinates and metadata
- API key authentication support

## Installation

```bash
pip install sh-ocr-client
```

## Requirements

- Python 3.9 or higher
- requests library (automatically installed as a dependency)

## Usage

```python
from sh_ocr_client import process_ocr

# Process an image
result = process_ocr(
    image_path="/path/to/your/image.png",
    language="ch",  # Language code: 'japan', 'ch', 'en', etc.
    api_key="your-api-key-here"
)

# The result is a dictionary containing:
# - success: Boolean indicating if processing was successful
# - data: Dictionary containing:
#   - text_blocks: List of extracted text blocks
#   - total_blocks: Number of text blocks found
#   - language: Detected/used language
print(result)
```

## Parameters

- `image_path` (str): Path to the image file to process
- `language` (str): Language code for OCR. Supported languages include:
  - `'ch'` - Chinese
  - `'japan'` - Japanese
  - `'en'` - English
  - And other language codes supported by the OCR service
- `api_key` (str): API key for authentication (required)

## Returns

Returns a dictionary with the following structure:

```python
{
    "success": True,
    "data": {
        "text_blocks": [...],  # List of text blocks with coordinates
        "total_blocks": 10,    # Number of text blocks
        "language": "ch"       # Language used
    }
}
```

## Error Handling

The function raises:
- `requests.RequestException`: If the request fails (network issues, etc.)
- `requests.HTTPError`: If the API returns an error status (e.g., 401 for invalid API key)

## Example

```python
from sh_ocr_client import process_ocr
import requests

try:
    result = process_ocr(
        image_path="menu.png",
        language="ch",
        api_key="sk-ocr-your-key-here"
    )
    
    if result.get('success'):
        text_blocks = result.get('data', {}).get('text_blocks', [])
        print(f"Found {len(text_blocks)} text blocks")
        for block in text_blocks:
            print(block.get('text', ''))
except requests.HTTPError as e:
    print(f"API Error: {e}")
except requests.RequestException as e:
    print(f"Request Error: {e}")
```

## License

MIT License

Copyright (c) 2025 shuaronhong

## Repository

https://github.com/shuaronhong/Pin_Hw09
