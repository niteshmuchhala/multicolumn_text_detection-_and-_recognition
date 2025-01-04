import pytesseract
from PIL import Image

def ocr_hindi_image(image_path):
    # Open the image file
    image = Image.open(image_path)
    
    # Perform OCR with Hindi language support
    text = pytesseract.image_to_string(image, lang='hin')
    
    return text

# Example usage
image_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\class_news_cutouts\WhatsApp Image 2024-09-09 at 11.33.55 PM.jpeg_cutout_3.jpg'
result = ocr_hindi_image(image_path)
print(result)