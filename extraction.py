from ultralytics import YOLO
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import yaml

# Load the trained YOLOv8 model
model_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\best (1).pt'  # Replace with your local path to the model weights
model = YOLO(model_path)

# Create the output directory if it doesn't exist
output_directory = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\class_news_cutouts'  # Replace with your local path
os.makedirs(output_directory, exist_ok=True)

# Define the class name for "news"
class_name = "news"

# Load the class names from the dataset YAML file
yaml_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\dataset\data.yaml'  # Replace with your local path to the YAML file
with open(yaml_path, 'r') as file:
    data = yaml.safe_load(file)
class_names = data['names']

# Find the index for the class name "news"
if class_name in class_names:
    news_class_index = class_names.index(class_name)
else:
    raise ValueError(f'Class name "{class_name}" not found in the dataset.')

# Specify the path to your single image
image_path = r'C:\Users\HP\OneDrive\Documents\newspaper_reading\WhatsApp Image 2024-09-09 at 11.33.55 PM.jpeg'  # Replace with your local image path

# Read the image
im = cv2.imread(image_path)

if im is None:
    print(f"Error: Couldn't read the image file {image_path}")
else:
    # Run prediction on the image
    results = model(im)

    # Get predictions
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes (xyxy format)
    classes = results[0].boxes.cls.cpu().numpy()  # Class indices
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Filter instances with the "news" class
    news_class_indices = np.where(classes == news_class_index)[0]  # Use the index for "news"

    if len(news_class_indices) > 0:
        # Show the original image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        plt.show()

        # Iterate over "news" class instances and cut out the bounding boxes
        for idx in news_class_indices:
            bbox = boxes[idx]
            x1, y1, x2, y2 = bbox
            cutout = im[int(y1):int(y2), int(x1):int(x2)]

            # Show the cutout
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB))
            plt.title(f'Cutout for "news" class instance {idx}')
            plt.axis('off')
            plt.show()

            # Save the cutout as a separate image
            output_image_path = os.path.join(output_directory, f"{os.path.basename(image_path)}_cutout_{idx}.jpg")
            cv2.imwrite(output_image_path, cutout)

            print(f"Saved cutout for 'news' class instance: {output_image_path}")
    else:
        print(f"No 'news' class found in image: {image_path}")
