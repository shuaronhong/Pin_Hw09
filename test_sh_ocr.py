from sh_ocr_client.client import process_ocr

if __name__ == "__main__":
    image_path = r"D:\Development\Pin_Hw\Google_Menu\back_end\images\Snipaste_03.png"
    language = 'ch'
    api_key = "sk-ocr-7f3a9b2c4d5e6f8a1b3c4d5e6f7a8b9c0"
    
    result = process_ocr(image_path, language, api_key)
    print(result)