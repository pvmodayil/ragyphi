#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 11-03-2025
# Topic         : YOLO based image extraction
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
from ultralytics import YOLO
from PIL import Image
from . import os, np

def pre_process_image(page_image: Image.Image) -> np.ndarray:
    YOLO_INPUT_SIZE: int = 640
    
    # Resize the image to 640x640 pixels
    resized_image: Image.Image = page_image.resize((YOLO_INPUT_SIZE,YOLO_INPUT_SIZE), Image.Resampling.LANCZOS)
    
    # Convert PIL to array (cv can only process arrays)
    image_array: np.ndarray = np.array(resized_image)
    
    return image_array

# Function to crop bounding boxes from the image
def crop_bounding_boxes(image: np.ndarray, 
                        boxes: np.ndarray, 
                        class_ids: np.ndarray, 
                        target_class: int =0) -> list[Image.Image]:
    cropped_images: list = []
    for box, class_id in zip(boxes, class_ids):
        if class_id == target_class:  # Check if the class matches the target class
            x_min, y_min, x_max, y_max = map(int, box)  # Convert coordinates to integers
            cropped_image: np.ndarray = image[y_min:y_max, x_min:x_max]  # Crop the region
            cropped_images.append(Image.fromarray(cropped_image))
    return cropped_images

def get_images(page_image: Image.Image) -> list[Image.Image]:
    # Resize the image to YOLO standard (640,640)
    image: np.ndarray = pre_process_image(page_image)
    
    # Load and predict with YOLO
    model_path: str = os.path.join(os.getcwd(),"ragyphi","weights","yolo_model.pt") # Edit later for package suitable paths
    model = YOLO(model_path) # Input the complete path string
    results: list = model(image)
    
    # Extract bounding boxes and class IDs from the results
    boxes: np.ndarray = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in [x_min, y_min, x_max, y_max] format
    class_ids: np.ndarray = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Crop only the diagrams
    cropped_images: list[Image.Image] = crop_bounding_boxes(image, boxes, class_ids, target_class=0)
    
    return cropped_images
