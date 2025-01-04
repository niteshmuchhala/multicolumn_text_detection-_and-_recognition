import os
import pytesseract
from PIL import Image, ImageDraw, ImageFont

def ocr_hindi_image_to_image(image_path, folder_path):
    # Create folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Open the image file
    image = Image.open(image_path)
    
    # Perform OCR with Hindi language support
    text = pytesseract.image_to_string(image, lang='hin')
    
    # Create a new image with a white background to write the text on
    new_image = Image.new('RGB', (800, 600), color=(255, 255, 255))
    draw = ImageDraw.Draw(new_image)
    
    # Define font (you can adjust the font size or use a custom Hindi font)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Split the text into multiple lines to fit on the image
    lines = text.split('\n')
    y_text = 10
    for line in lines:
        draw.text((10, y_text), line, font=font, fill=(0, 0, 0))
        y_text += 30  # Adjust the line height
    
    # Save the image with the text overlaid inside the folder
    output_image_path = os.path.join(folder_path, 'ocr_output_image.jpg')
    new_image.save(output_image_path)
    
    return text

# Example usage
image_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\class_news_cutouts\WhatsApp Image 2024-09-09 at 11.33.55 PM.jpeg_cutout_3.jpg'
folder_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\ocr_images'

result = ocr_hindi_image_to_image(image_path, folder_path)
print(result)
