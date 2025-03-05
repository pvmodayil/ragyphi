#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 28.02.2025
# Topic         : Document pre process
# Documents     : pdf
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
# global imports
from . import os, pd, np
import gc

# pdf
from .contextualizer import LMContextualizer, VLMContextualizer
import uuid
import pdfplumber
from PIL import Image
from tqdm import tqdm

#####################################################################################
#                                    Functions
#####################################################################################
def saveData(path: str, data: str) -> None:
    """
    Open file and write text data
    
    Parameters
    ----------
    path : str
        file path
    data : str
        text data
    
    Raises
    ------
    Exception
       OSError if the path string contains unaccepted characters when trying to save data locally
    """
    try:
        with open(path, 'w') as f:
            f.write(data)
    except OSError:
        print(f"Something wrong with file name: {path}")

def extractText(page: pdfplumber.page.Page,
                base_dir: str,
                pdf_path:str,
                llm_model: LMContextualizer,
                extracted_items: list[dict]) -> list[dict]:
    # Get unique identifiers for the page
    ###########################################
    page_number: int = page.page_number
    pdf_filename: str = os.path.splitext(os.path.basename(pdf_path))[0]
    
    page_content_text: str = page.extract_text()
    
    # Summarize and store in structured format
    extracted_items.append({
                "uuid": str(uuid.uuid4()),
                "text": llm_model.contextualizeDataWithLM(content_to_summarize=f"Page text:\n{page_content_text}"),
                "metadata":{
                    "file": pdf_filename,
                    "page": page_number, 
                    "type": "text",
                    "original_content": page_content_text}
                })
    # Save text
    saveData(path=os.path.join(base_dir,"text",f"{pdf_filename}_{page_number}_text.txt"), 
            data=page_content_text)
    saveData(path=os.path.join(base_dir,"text",f"{pdf_filename}_{page_number}_context.md"), 
            data=extracted_items[-1]["text"])
    
    return extracted_items

def extractTable(page: pdfplumber.page.Page,
                base_dir: str,
                pdf_path:str,
                llm_model: LMContextualizer,
                extracted_items: list[dict]) -> list[dict]:
    # Get unique identifiers for the page
    ###########################################
    page_number: int = page.page_number
    pdf_filename: str = os.path.splitext(os.path.basename(pdf_path))[0]
    
    page_content_text: str = page.extract_text() # need it for contextualizing
    
    page_content_tables:list[list[list]] = page.extract_tables()
    for table_id,table in enumerate(page_content_tables):
        # Convert extracted table to pandas dataframe
        df: pd.DataFrame = pd.DataFrame(table[1:], columns=table[0])  # Create DataFrame
        table_content_text: str = df.to_markdown()
        
        # Summarize and store in structured format
        extracted_items.append({
                    "uuid": str(uuid.uuid4()), 
                    "text": llm_model.contextualizeDataWithLM(
                        content_to_summarize=f"Page text:\n{page_content_text}\nTable:\n{table_content_text}"),
                    "metadata":{
                        "file": pdf_filename,
                        "page": page_number,
                        "type": "table",
                        "original_content": table_content_text}
                    })

        # Save table
        table_filename: str = os.path.join(base_dir,"tables",f"{pdf_filename}_{page_number}_table_{table_id}.csv")  
        saveData(path=os.path.join(base_dir,"tables",f"{pdf_filename}_{page_number}_context_{table_id}.md"), 
                data=extracted_items[-1]["text"])
        try:
            df.to_csv(table_filename,index=False)
        except OSError:
            print(f"Something wrong with file name: {table_filename}")
            
        return extracted_items

def mergeBBOXES(bboxes: list[tuple]):
    if not bboxes:
        return []
    
    bboxes.sort(key=lambda b: b[1])  # Sort bboxes by top-left corner's y-coordinate

    merged_bboxes = []
    current_bbox = list(bboxes[0]) # Changing to list to allow mutability

    # Bbox tuple structure: (x1,y1,x2,y2) => x1,y1: top left , x2,y2,: bottom right
    for bbox in bboxes[1:]:
        # x11 <= x22 and x12 >= x21 and y11 <= y22 and y12 >= y21
        if (current_bbox[0] <= bbox[2] and current_bbox[2] >= bbox[0] and 
                current_bbox[1] <= bbox[3] and current_bbox[3] >= bbox[1]):
            current_bbox[0] = min(current_bbox[0], bbox[0])
            current_bbox[1] = min(current_bbox[1], bbox[1])
            current_bbox[2] = max(current_bbox[2], bbox[2])
            current_bbox[3] = max(current_bbox[3], bbox[3])
            # Continue the search until a separate bbox is found
        else:
            merged_bboxes.append(tuple(current_bbox)) # Add the merged bbox
            current_bbox = list(bbox) # Shift the check to the next separate bbox

    merged_bboxes.append(tuple(current_bbox)) # Add the merged bbox
    
    return merged_bboxes

def processImageWithCV(base_dir: str,
                       pdf_filename: str,
                       page_number: int,
                       page_image_PIL: Image.Image,
                       contour_threshold: int | None = 400) -> list[Image.Image] :
    import cv2
    # Convert PIL to array (cv can only process arrays)
    image: np.ndarray = np.array(page_image_PIL)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to hold potential images
    potential_image_contours: list = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > contour_threshold:  # Minimum area threshold; adjust as needed
            x, y, w, h = cv2.boundingRect(contour)
            potential_image_contours.append((x, y, x + w, y + h))
    
    # If there are any potential images
    cropped_images: list = []
    if potential_image_contours: 
        merged_bboxes: list[tuple] = mergeBBOXES(potential_image_contours)
        
        # Crop out images
        for (x1, y1, x2, y2) in merged_bboxes:
            # Extract the contour region from image array
            cropped_image: np.ndarray = image[y1:y2, x1:x2]
            cropped_images.append(Image.fromarray(cropped_image)) # Append to return list
    
    return cropped_images   
     
def extractImage(page: pdfplumber.page.Page,
                base_dir: str,
                pdf_path:str,
                vllm_model: VLMContextualizer,
                extracted_items: list[dict]) -> list[dict]:
    # Get unique identifiers for the page
    ###########################################
    page_number: int = page.page_number
    pdf_filename: str = os.path.splitext(os.path.basename(pdf_path))[0]
    page_content_text: str = page.extract_text() # need it for contextualizing
    
    # Convert the whole page into a PIL image
    page_image: pdfplumber.display.PageImage = page.to_image() 
    extracted_images: list[Image.Image] = processImageWithCV(base_dir,
                                                             pdf_filename,
                                                             page_number,
                                                             page_image.original)
     
    if extracted_images:
        for img_id, image in enumerate(extracted_images):
            img_save_path = os.path.join(base_dir,"images",f"{pdf_filename}_{page_number}_image_{img_id}.jpg")
            image.save(img_save_path)
            # Summarize and store in structured format
            extracted_items.append({
                        "uuid": str(uuid.uuid4()), 
                        "text": vllm_model.contextualizeDataWithVLM(additional_text=page_content_text,image=image),
                        "metadata":{
                            "file": pdf_filename,
                            "page": page_number,
                            "type": "image",
                            "original_content": img_save_path}
                        })
            saveData(path=os.path.join(base_dir,"images",f"{pdf_filename}_{page_number}_context_{img_id}.md"), 
                data=extracted_items[-1]["text"])

    return extracted_items
              
def processPDF(base_dir: str,
               pdf_path:str,
               extracted_items: list[dict],
               local_llm: str,
               local_vllm: str) -> list[dict]:
    """
    Takes pdf file and extracts text, tables and images which are summarized along with context using LLM model.
    
    Parameters
    ----------
    base_dir : str
        base directory path
    pdf_path : str
        path string to the pdf file being analyzed
    extracted_items : list[dict]
        list of dictionaries with the below format
        {
         "uuid": unique id,
         "text": generated summary,
         "metadata":{
             "file": name of the document,
             "page": page number, 
             "type": type of the extarcted data,
             "original_content": original content which was summarized
              }
        } 
    local_llm : str 
        Ollama chat model name
    local_Vllm : str
        Ollama vision model name

    Returns
    -------
    list[dict]
        extracted data 
    """
    with pdfplumber.open(pdf_path) as pdf: # Memory will be removed once exited
        for page in tqdm(pdf.pages, desc=f"Processing {pdf_path} pages"):
            # Initialize the LLM model
            ###########################################
            llm_model = LMContextualizer(local_llm=local_llm)  
            
            # Extract text and contextualize
            ###########################################
            extracted_items = extractText(page,base_dir,pdf_path,llm_model,extracted_items)
            
            # Extract tables and contextualize
            #########################################
            extracted_items = extractTable(page,base_dir,pdf_path,llm_model,extracted_items)
            
            # Initialize the Vision LLM model
            ##########################################
            del llm_model # delete the loaded llm model
            vllm_model = VLMContextualizer(loacal_vllm=local_vllm)
            
            # Extract images and contextualize
            #########################################
            extracted_items = extractImage(page,base_dir,pdf_path,vllm_model,extracted_items)
            
            # Free memory
            del vllm_model
    print(f"Texts and tables with context are extracted from file {pdf_path}")
    
    
    gc.collect        
    
    return extracted_items