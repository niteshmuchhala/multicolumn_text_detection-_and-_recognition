from ultralytics import YOLO
import cv2
import numpy as np
import os
import yaml
import pytesseract
from PIL import Image
from gtts import gTTS
from playsound import playsound
from matplotlib import pyplot as plt

def extract_news_articles(model, image_path, output_directory, class_name, class_names):
    if class_name not in class_names:
        raise ValueError(f'Class name "{class_name}" not found in the dataset.')
    news_class_index = class_names.index(class_name)

    # Load the image
    im = cv2.imread(image_path)
    if im is None:
        raise FileNotFoundError(f"Error: Couldn't read the image file {image_path}")

    # Run prediction
    results = model(im)

    # Get predictions
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (xyxy format)
    classes = results[0].boxes.cls.cpu().numpy()  # Class indices
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Filter instances for the "news" class
    news_class_indices = np.where(classes == news_class_index)[0]

    cutout_paths = []
    for idx in news_class_indices:
        bbox = boxes[idx]
        x1, y1, x2, y2 = bbox
        cutout = im[int(y1):int(y2), int(x1):int(x2)]

        # Save the cutout
        output_image_path = os.path.join(output_directory, f"cutout_{idx}.jpg")
        cv2.imwrite(output_image_path, cutout)
        cutout_paths.append(output_image_path)

    # Display the input image
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Input Newspaper Image")
    plt.axis('off')
    plt.show()

    return cutout_paths

def ocr_hindi_image(image_path):
    # Perform OCR with Hindi language support
    text = pytesseract.image_to_string(Image.open(image_path), lang='hin')
    return text

def convert_text_to_audio(text, output_audio_path):
    # Convert text to audio using gTTS
    tts = gTTS(text=text, lang='hi')
    tts.save(output_audio_path)

def process_newspaper_image(model_path, yaml_path, input_image_path, output_base_path, class_name="news"):
    # Load the YOLO model
    model = YOLO(model_path)

    # Load class names from the dataset YAML file
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    class_names = data['names']

    # Create necessary directories
    cutouts_dir = os.path.join(output_base_path, 'cutouts')
    ocr_results_dir = os.path.join(output_base_path, 'ocr_results')
    audio_results_dir = os.path.join(output_base_path, 'audio_results')
    os.makedirs(cutouts_dir, exist_ok=True)
    os.makedirs(ocr_results_dir, exist_ok=True)
    os.makedirs(audio_results_dir, exist_ok=True)

    # Extract news articles
    cutout_paths = extract_news_articles(model, input_image_path, cutouts_dir, class_name, class_names)
    print(f"Extracted {len(cutout_paths)} news articles.")

    # Process each cutout
    for idx, cutout_path in enumerate(cutout_paths):
        print(f"\nProcessing Cutout {idx + 1}...")
        
        # Perform OCR
        ocr_text = ocr_hindi_image(cutout_path)
        ocr_output_text_path = os.path.join(ocr_results_dir, os.path.basename(cutout_path).replace('.jpg', '.txt'))
        with open(ocr_output_text_path, 'w', encoding='utf-8') as f:
            f.write(ocr_text)

        # Convert text to audio
        audio_output_path = os.path.join(audio_results_dir, os.path.basename(cutout_path).replace('.jpg', '.mp3'))
        convert_text_to_audio(ocr_text, audio_output_path)

        # Display the cutout image
        cutout_image = cv2.imread(cutout_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(cutout_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Cutout {idx + 1}")
        plt.axis('off')
        plt.show()

        # Print OCR text
        print(f"OCR Text for Cutout {idx + 1}:")
        print(ocr_text)

        # Play the audio
        print(f"Playing audio for Cutout {idx + 1}...")
        playsound(audio_output_path)

        print(f" - OCR text saved to: {ocr_output_text_path}")
        print(f" - Audio saved to: {audio_output_path}")

# Example Usage
if __name__ == "__main__":
    model_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\best (1).pt'
    yaml_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\dataset\data.yaml'
    input_image_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\WhatsApp Image 2024-09-09 at 11.33.55 PM.jpeg'
    output_base_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading'

    process_newspaper_image(model_path, yaml_path, input_image_path, output_base_path)
