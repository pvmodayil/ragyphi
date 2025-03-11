#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 11-03-2025
# Topic         : YOLO based image extraction
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

def pre_process_image(page_image: Image.Image) -> np.ndarray:
    YOLO_INPUT_SIZE: int = 640
    
    # Convert PIL to array (cv can only process arrays)
    image: np.ndarray = np.array(page_image)
    
    # If the image has an alpha channel (RGBA), convert it to BGR
    if image.shape[-1] == 4:  # Check if the image has 4 channels
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize the image to 640x640 pixels
    resized_image: np.ndarray = cv2.resize(image, (YOLO_INPUT_SIZE,YOLO_INPUT_SIZE))
    
    return resized_image

# Function to crop bounding boxes from the image
def crop_bounding_boxes(image: np.ndarray, 
                        boxes: np.ndarray, 
                        class_ids: np.ndarray, 
                        target_class: int =0) -> list[Image.Image]:
    cropped_images: list = []
    for box, class_id in zip(boxes, class_ids):
        if class_id == target_class:  # Check if the class matches the target class
            x_min, y_min, x_max, y_max = map(int, box)  # Convert coordinates to integers
            cropped_image = image[y_min:y_max, x_min:x_max]  # Crop the region
            cropped_images.append(Image.fromarray(cropped_image))
    return cropped_images

def get_images(page_image: Image.Image) -> list[Image.Image]:
    # Resize the image to YOLO standard (640,640)
    image: np.ndarray = pre_process_image(page_image)
    
    # Load and predict with YOLO
    model = YOLO("yolo_model.pt")
    results = model(image)
    
    # Extract bounding boxes and class IDs from the results
    boxes: np.ndarray = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in [x_min, y_min, x_max, y_max] format
    class_ids: np.ndarray = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Crop only the diagrams
    cropped_images: list[Image.Image] = crop_bounding_boxes(image, boxes, class_ids, target_class=0)
    
    return cropped_images
