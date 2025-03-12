#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 28.02.2025
# Topic         : Document pre process
# Documents     : pdf,
#################################################
"""
To Do: 
    Add multi document type processing capabilities
"""
#####################################################################################
#                                     Imports
#####################################################################################
# Global imports
from . import os, pd, np

# Database
import faiss

# Text extraction
from langchain_huggingface import HuggingFaceEmbeddings

# Typing
from ._types import ExtractedItems
        
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
    directories: list[str] = ["images", "text","tables"]
    for dir in directories:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)
        
###########################################
#        Document Processing: PDF
###########################################   
def loadFilesAndExtract(base_dir: str, 
                      extracted_items: list[ExtractedItems], 
                      local_llm: str,
                      local_vllm: str) -> list[ExtractedItems]:
    """
    Load the file types within the given base_dir path and extract data from them

    Parameters
    ----------
    base_dir : str
        base directory path
    extracted_items : list[ExtractedItems]
        list to store the extracted data
    local_llm : str
        Ollama chat model name default value llama3.2-8b model
    local_vllm : str
        Ollama vision model name default value granite3.2-vision model

    Returns
    -------
    list[ExtractedItems]
        extracted data
    """
    from .pdf_extractor import processPDF
    
    for file in os.listdir(base_dir):
        if file.endswith('.pdf'):
            pdf_path: str = os.path.join(base_dir, file)
            extracted_items = processPDF(base_dir=base_dir,
                                         pdf_path=pdf_path,
                                         extracted_items=extracted_items,
                                         local_llm=local_llm,
                                         local_vllm=local_vllm)      
        
    return extracted_items   

def convertExtractedItemsToDF(extracted_items: list[ExtractedItems]) -> pd.DataFrame:
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
        all_embeddings: np.ndarray = np.array(extracted_data_df['embedding'].tolist(), dtype=np.float32)
    except KeyError:
        raise Exception("Did not find the column 'embedding' in the extracted_data dataframe.")
    except ValueError:
        raise Exception("Failed to convert embeddings to a 2D numpy array. Check if all embeddings have the same dimension.")
    
    # Define the dimension of your embedding vectors
    embedding_vector_dimension: int = all_embeddings.shape[1]

    # Create FAISS Index
    index: faiss.IndexFlatL2 = faiss.IndexFlatL2(embedding_vector_dimension)

    # Clear any pre-existing index
    index.reset()

    # Verify the array properties
    assert isinstance(all_embeddings, np.ndarray), f"all_embeddings must be a numpy array, got type: {type(all_embeddings)}"
    assert all_embeddings.ndim == 2, f"all_embeddings must be a 2D array, got dimension: {all_embeddings.ndim}"
    assert all_embeddings.size > 0, "all_embeddings must not be empty."
    assert all_embeddings.dtype == np.float32, f"Expected dtype np.float32, but got {all_embeddings.dtype}"
    
    # Add embeddings to the index
    index.add(np.array(all_embeddings, dtype=np.float32)) # type: ignore => FAISS not type check ready https://github.com/facebookresearch/faiss/issues/2891
    
    # Create vector store directory
    os.makedirs("vector_store",exist_ok=True)
    vector_store_filename: str = os.path.join(os.getcwd(),"vector_store",'faiss_index.index')

    # Save the vector embeddings
    faiss.write_index(index, vector_store_filename)
    print(f"FAISS Index saved at {vector_store_filename}")
    
    # Save rest of the extracted data
    extracted_data_df.to_parquet(os.path.join(os.getcwd(),"vector_store",'extracted_data.parquet'), 
                                 compression='snappy')

def checkFolder(dir_path: str) -> None:
    dir_contents: list[str] = os.listdir(dir_path)
    files: list[str] = [file for file in dir_contents if os.path.isfile(os.path.join(dir_path,file))]
    
    if not files:
        raise Exception(f"The {dir_path} folder is empty. Please provide relevant documents for RAG")

def processDocuments(local_llm: str = "llama3.2:3b-instruct-fp16",
                     local_vllm: str = "granite3.2-vision:latest") -> None:
    """
    Create a data folder and process the files within it.

    Parameters
    ----------
    local_llm : str, optional
        name of llm model, by default "llama3.2:3b-instruct-fp16"
    local_vllm : str, optional
        name of vllm model, by default "granite3.2-vision:latest"
        
    Raises
    ------
    Exception
        ResponseError if the ollama model is not found, try ollama pull model-name
    """
    print("Creating directory data.......\n Please include any documents that need to be extracted if they are not already provided.")
    # Add more supported document type later
    print("Supported document types:\n- PDF")
    
    # Create data directory and check contents
    ###########################################
    os.makedirs("data", exist_ok=True)
    base_dir: str = os.path.join(os.getcwd(),"data")
    checkFolder(base_dir)
    
    create_directories(base_dir)
     
    # Extract information from pdf files
    ###########################################
    extracted_items: list[ExtractedItems] = []
    extracted_items = loadFilesAndExtract(base_dir=base_dir,
                                          extracted_items=extracted_items,
                                          local_llm=local_llm,
                                          local_vllm=local_vllm)
    
    # Convert to dataframe
    ###########################################
    extracted_data_df: pd.DataFrame = convertExtractedItemsToDF(extracted_items)
    
    # Embed extracted text
    ###########################################
    extracted_data_df = embedExractedData(extracted_data_df)
    
    # Index with FAISS and store data
    indexAndStore(extracted_data_df)