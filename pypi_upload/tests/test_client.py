import sys
import os
import json

# Add the src directory to the path so we can import the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sh_ocr_client.client import process_ocr


def test_process_ocr():
    """Test the process_ocr function with a sample image."""
    image_path = r"D:\Development\Pin_Hw\Google_Menu\back_end\images\Snipaste_03.png"
    language = 'ch'
    api_key = "sk-ocr-7f3a9b2c4d5e6f8a1b3c4d5e6f7a8b9c0"
    
    result = process_ocr(image_path, language, api_key)
    print(result)
    
    # Assert that the result is a dictionary
    assert isinstance(result, dict), "Result should be a dictionary"
    
    # Assert that the response indicates success
    assert result.get('success') == True, "OCR processing should be successful"
    
    # Assert that data exists
    assert 'data' in result, "Response should contain 'data' field"
    
    # Assert that text_blocks exist in the data
    assert 'text_blocks' in result.get('data', {}), "Data should contain 'text_blocks'"
    
    print(f"Test passed! Found {result.get('data', {}).get('total_blocks', 0)} text blocks.")
    print(f"Language: {result.get('data', {}).get('language')}")
    
    return result


if __name__ == '__main__':
    test_process_ocr()

