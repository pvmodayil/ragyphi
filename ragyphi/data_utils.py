#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 19.02.2025
# Topic         : Document pre process
# Documents     : pdf,
#################################################
"""
To Do: 
    Add multi document processing capabilities
    Add Image processing capabilities
"""
#####################################################################################
#                                     Imports
#####################################################################################
# Global imports
from . import os, pd, np

# Memory management
import gc

# pdf
import pdfplumber
import uuid

# Database
import faiss

# Text extraction
from langchain_huggingface import HuggingFaceEmbeddings

# LLM
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

############################
# Custom Exception
############################
class ModelNotFoundError(Exception):
    """Custom exception to handle cases where the specified model cannot be found."""
    def __init__(self, model_name):
        super().__init__(f"Model '{model_name}' not found. Please pull it first using 'ollama pull {model_name}'.")
        
#####################################################################################
#                                    Functions
#####################################################################################
# Create the directories
def create_directories(base_dir: str) -> None:
    """
    Takes in a base directory path and creates required folders in it
    Parameters
    ----------
    base_dir : str
        base directory path

    Returns
    -------
    None
        return nothing, creates the directories
    """
    directories: list = ["images", "text","tables"]
    for dir in directories:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)
        
###########################################
#        Document Processing: PDF
###########################################   
def contextualizeData(content_to_summarize: str,
                      llm: ChatOllama) -> str:
    """
    Takes in text to be summarised and summarises it.
    Parameters
    ----------
    content_to_summarize : str
        text / table along with context that needs to be summarised 
    llm: ChatOllama 
        Ollama chat model default value llama3.2-8b model

    Returns
    -------
    str
        summarised text 
    """
    # Contextualize prompt
    ###############################################
    contextualizerInstruction: str = """You are a helpful assistant capable of summarizing texts and tables for retrieval."""
    
    contextualizerPrompt: str = """ Carefully analyse the text or table data from the document and provide a detailed summary.\
    These summaries will be embedded and used to retrieve the raw text or table elements.\
    Also generate hypothetical questions that can be answered based on the the given context.

    Document to be summarized:
    {content_to_summarize}

    Please structure your response in the following format:
    1. A concise summary of the table or text that is well optimized for retrieval.
    2. List the key observations and relevant metrics.
    3. List of the major keywords.
    4. A list of exactly 3 hypothetical questions that the above document could be used to answer.
    """
    
    return llm.invoke(
    [SystemMessage(content=contextualizerInstruction)]
    + [HumanMessage(content=contextualizerPrompt.format(content_to_summarize=content_to_summarize))]
    ).content

def saveData(path: str, data: str) -> None:
    """
    Open file and write text data
    Parameters
    ----------
    path : str
        file path
    data : str
        text data
    """
    try:
        with open(path, 'w') as f:
            f.write(data)
    except OSError:
        print(f"Something wrong with file name: {path}")
                
                
def extractAndContextualizePDFPage(base_dir: str, 
                      pdf_path: str, 
                      page: pdfplumber.page.Page,  
                      extracted_items: list[dict],
                      llm: ChatOllama) -> list[dict]:
    """
    Takes in pdf pages and extracts text, tables and summarize each along with context using LLM model.
    Parameters
    ----------
    base_dir : str
        base directory path
    pdf_path : str
        path string to the pdf file being analyzed
    page: pdfplumber.page.Page
        pdfplumber library page object
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
    llm: ChatOllama 
        Ollama chat model default value llama3.2-8b model

    Returns
    -------
    list[dict]
        extracted data 
    
    Raises
    ------
    Exception
       OSError if the path string contains unaccepted characters when trying to save data locally
    """
    # Get unique identifiers for the page
    ###########################################
    page_number: int = page.page_number
    pdf_filename: str = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Extract text and contextualize
    ###########################################
    page_content_text: str = page.extract_text()
    
    # Summarize and store in structured format
    extracted_items.append({
                "uuid": str(uuid.uuid4()),
                "text": contextualizeData(content_to_summarize=f"Page text:\n{page_content_text}",llm=llm),
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
    
    # Extract tables and contextualize
    #########################################
    page_content_tables:list[list[list]] = page.extract_tables()
    for table_id,table in enumerate(page_content_tables):
        # Convert extracted table to pandas dataframe
        df: pd.DataFrame = pd.DataFrame(table[1:], columns=table[0])  # Create DataFrame
        table_content_text: str = df.to_markdown()
        
        # Summarize and store in structured format
        extracted_items.append({
                    "uuid": str(uuid.uuid4()), 
                    "text": contextualizeData(content_to_summarize=f"Page text:\n{page_content_text}\nTable:\n{table_content_text}",llm=llm),
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

def loadFileAndExtract(base_dir: str, 
                      extracted_items: list[dict], 
                      llm: ChatOllama) -> list[dict]:
    """
    Load the file types within the given base_dir path and extract data from them

    Parameters
    ----------
    base_dir : str
        base directory path
    extracted_items : list[dict]
        list to store the extracted data
    llm : ChatOllama
        Ollama chat model default value llama3.2-8b model

    Returns
    -------
    list[dict]
        extracted data
    """
    from tqdm import tqdm
    
    for file in os.listdir(base_dir):
        if file.endswith('.pdf'):
            pdf_path: str = os.path.join(base_dir, file)
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in tqdm(pdf.pages, desc=f"Processing {pdf_path} pages"):
                    # Extract context
                    extracted_items: list[dict] = extractAndContextualizePDFPage(base_dir, pdf_path, page, extracted_items, llm)
            
            print(f"Texts and tables with context are extracted from file {pdf_path}")       
        
    return extracted_items   

def convertExtractedItemsToDF(extracted_items: list[dict]) -> pd.DataFrame:
    """
    Convert the extracted_items list of dictionaries to pandas dataframe

    Parameters
    ----------
    extracted_items : list[dict]
        extracted data

    Returns
    -------
    pd.DataFrame
        dataframe
    """
    # Convert the list of dictionaries into a single dataframe
    return pd.DataFrame(extracted_items)

###########################################
#        Embed and Store Data
###########################################  
def embedExractedData(extracted_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Embed the text(contextualized) column in the dataframe

    Parameters
    ----------
    extracted_data_df : pd.DataFrame
        extracted data

    Returns
    -------
    pd.DataFrame
        datframe with embeddings

    Raises
    ------
    Exception
       KeyError if the column 'text' is not present 
    """
    # Embedding model
    embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        # The extracted_data_df has column "text" which contains the text chunks to be embedded
        print("Embedding text chunks..........")
        extracted_data_df['embedding'] = embedding_model.embed_documents(extracted_data_df["text"].tolist())
    except KeyError:
        raise Exception("Did not find column 'text' in the extracted_data dataframe.\nMake sure extractable text is present in the document.")
    
    return extracted_data_df

def indexAndStore(extracted_data_df: pd.DataFrame) -> None:
    """
    Create a FAISS vector store and save both vector store and extracted data locally

    Parameters
    ----------
    extracted_data_df : pd.DataFrame
        extracted data

    Raises
    ------
    Exception
        KeyError if column 'embedding' is not in the dataframe
    """
    print("Storing vectors and metadata..........")
    try:
        # Convert the embeddings(array) to an array of arrays
        all_embeddings: np.ndarray = np.array(extracted_data_df['embedding'].tolist(), dtype='object')
    except KeyError:
        raise Exception("Did not find the column 'embedding' in the extracted_data dataframe.")
    
    # Define the dimension of your embedding vectors
    embedding_vector_dimension: int = all_embeddings.shape[1]

    # Create FAISS Index
    index: faiss.IndexFlatL2 = faiss.IndexFlatL2(embedding_vector_dimension)

    # Clear any pre-existing index
    index.reset()

    # Add embeddings to the index
    index.add(np.array(all_embeddings, dtype=np.float32))
    
    # Create vector store directory
    os.makedirs("vector_store",exist_ok=True)
    vector_store_filename = os.path.join(os.getcwd(),"vector_store",'faiss_index.index')

    # Save the vector embeddings
    faiss.write_index(index, vector_store_filename)
    print(f"FAISS Index saved at {vector_store_filename}")
    
    # Save rest of the extracted data
    extracted_data_df.to_parquet(os.path.join(os.getcwd(),"vector_store",'extracted_data.parquet'), 
                                 compression='snappy')

def processDocuments(local_llm: str = "llama3.2:3b-instruct-fp16") -> None:
    """
    Create a data folder and process the files within it.

    Parameters
    ----------
    local_llm : str, optional
        name of llm model, by default "llama3.2:3b-instruct-fp16"
        
    Raises
    ------
    Exception
        ResponseError if the ollama model is not found, try ollama pull model-name
    """
    print("Creating directory data.......\n Please include any documents that need to be extracted if they are not already provided.")
    # Add more supported document type later
    print("Supported document types:\n- PDF")
    
    # Create data directory
    ###########################################
    os.makedirs("data", exist_ok=True)
    base_dir: str = os.path.join(os.getcwd(),"data")
    create_directories(base_dir)
    
    # Load llm
    ###########################################
    try:
        llm: ChatOllama = ChatOllama(model=local_llm, temperature=0)
    except Exception as e:
        if "404" in str(e):
            raise ModelNotFoundError(local_llm)
        else: 
            print(f"Unexpected error occured: {e}")
            import sys
            sys.exit(1) # exit the program
     
    # Extract information from pdf files
    ###########################################
    extracted_items: list = []
    extracted_items = loadFileAndExtract(base_dir,extracted_items,llm)
    
    # Convert to dataframe
    ###########################################
    extracted_data_df: pd.DataFrame = convertExtractedItemsToDF(extracted_items)
    
    # Embed extracted text
    ###########################################
    extracted_data_df = embedExractedData(extracted_data_df)
    
    # Index with FAISS and store data
    indexAndStore(extracted_data_df)
    
    # Free memory
    
    del llm
    gc.collect()