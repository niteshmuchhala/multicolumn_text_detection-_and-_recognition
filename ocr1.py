import os
import pytesseract
from PIL import Image

def ocr_hindi_image(image_path, folder_path):
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Open the image file
    image = Image.open(image_path)
    
    # Perform OCR with Hindi language support
    text = pytesseract.image_to_string(image, lang='hin')
    
    # Save the OCR result to a text file inside the folder
    output_text_path = os.path.join(folder_path, 'ocr_output.txt')
    with open(output_text_path, 'w', encoding='utf-8') as file:
        file.write(text)
    
    return text

# Example usage
image_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\class_news_cutouts\WhatsApp Image 2024-09-09 at 11.38.20 PM.jpeg_cutout_2.jpg'
folder_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\ocr_results'

result = ocr_hindi_image(image_path, folder_path)
print(result)
