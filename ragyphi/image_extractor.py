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

def preProcessImage(page_image: Image.Image) -> Image.Image:
    YOLO_INPUT_SIZE: int = 640
    
    # Resize the image to 640x640 pixels
    resized_image: Image.Image = page_image.resize((YOLO_INPUT_SIZE,YOLO_INPUT_SIZE), Image.Resampling.LANCZOS)
    
    return resized_image

# Function to crop bounding boxes from the image
def cropBoundingBoxes(original_image: Image.Image, 
                        boxes: np.ndarray, 
                        class_ids: np.ndarray, 
                        target_class: int =0) -> list[Image.Image]:
    # resized image resolution
    YOLO_INPUT_SIZE: int = 640
    original_width: int = original_image.size[0]
    original_height: int = original_image.size[1]
    
    cropped_images: list = []
    for box, class_id in zip(boxes, class_ids):
        if class_id == target_class:  # Check if the class matches the target class
            # Convert coordinates to integers
            x_min, y_min, x_max, y_max = map(int, box)  
            
            # Calculate the scaling factor
            scale_x: float = original_width / YOLO_INPUT_SIZE
            scale_y: float = original_height / YOLO_INPUT_SIZE
            
            # Apply the scaling factor to the bounding box coordinates
            x_min_scaled = int(x_min * scale_x)
            y_min_scaled = int(y_min * scale_y)
            x_max_scaled = int(x_max * scale_x)
            y_max_scaled = int(y_max * scale_y)
            
            # Crop the region from the original image
            cropped_image: Image.Image = original_image.crop((x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled))
            cropped_images.append(cropped_image)
    
    return cropped_images

def getImages(page_image: Image.Image) -> list[Image.Image]:
    # Resize the image to YOLO standard (640,640)
    image: Image.Image = preProcessImage(page_image)
    
    # Load and predict with YOLO
    model_path: str = os.path.join(os.getcwd(),"ragyphi","weights","yolo_model.pt") # Edit later for package suitable paths
    model = YOLO(model_path) # Input the complete path string
    results: list = model(image)
    
    # Extract bounding boxes and class IDs from the results
    boxes: np.ndarray = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in [x_min, y_min, x_max, y_max] format
    class_ids: np.ndarray = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Crop only the diagrams
    cropped_images: list[Image.Image] = cropBoundingBoxes(page_image, boxes, class_ids, target_class=0)
    
    return cropped_images
