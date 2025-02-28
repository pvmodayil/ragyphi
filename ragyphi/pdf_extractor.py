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
from . import os, pd
import gc

# pdf
from .contextualizer import Contextualizer
import uuid
import pdfplumber
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
                llm_model: Contextualizer,
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
                llm_model: Contextualizer,
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
                data=page_content_text)
        try:
            df.to_csv(table_filename,index=False)
        except OSError:
            print(f"Something wrong with file name: {table_filename}")
            
        return extracted_items
            
def processPDF(base_dir: str,
               pdf_path:str,
               extracted_items: list[dict],
               local_llm: str) -> list[dict]:
    """
    Takes pdf file and extracts text, tables and images which are summarized along with context using LLM model.
    
    Parameters
    ----------
    base_dir : str
        base directory path
    pdf_path : str
        path string to the pdf file being analyzed
    extracted_items: list[dict]
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
    local_llm: str 
        Ollama chat model name

    Returns
    -------
    list[dict]
        extracted data 
    """
    # Initialize the LLM model
    llm_model = Contextualizer(local_llm=local_llm)      
    
    with pdfplumber.open(pdf_path) as pdf: # Memory will be removed once exited
        for page in tqdm(pdf.pages, desc=f"Processing {pdf_path} pages"):
            # Extract text and contextualize
            ###########################################
            extracted_items = extractText(page,base_dir,pdf_path,llm_model,extracted_items)
            
            # Extract tables and contextualize
            #########################################
            extracted_items = extractTable(page,base_dir,pdf_path,llm_model,extracted_items)
            
            # Extract images and contextualize
            #########################################
    print(f"Texts and tables with context are extracted from file {pdf_path}")
    
    # Free memory
    del llm_model
    gc.collect        
    
    return extracted_items