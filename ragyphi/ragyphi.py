#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 28.02.2025
# Topic         : RAG application
#################################################

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
from dataclasses import dataclass

# LLM
from .ollama_chat import Chatbot
# from ollama._types import ResponseError

# Prompt
from .prompt_template import getUserPrompt

############################
# Custom Exception
############################
class ModelNotFoundError(Exception):
    """Custom exception to handle cases where the specified model cannot be found."""
    def __init__(self, model_name: str) -> None:
        super().__init__(f"Model '{model_name}' not found. Please pull it first using 'ollama pull {model_name}'.")


############################
# Custom DataClasses
############################
@dataclass(frozen=True)
class ExtractedData:
    index: faiss.IndexFlatL2
    raw_data: pd.DataFrame

@dataclass(frozen=True)
class LLMResponse:
    response: str
    context: str
   
###########################################
#        Load and Retrieve
###########################################      
def loadData() -> ExtractedData:
    """
    Load the vector store and extracted data, if not found call processDocuments function

    Returns
    -------
    ExtractedData
        stored vector store and extracted data
    """
    vector_store_filename: str = os.path.join(os.getcwd(),"vector_store",'faiss_index.index')
    extracted_data_filename: str = os.path.join(os.getcwd(),"vector_store",'extracted_data.parquet')
    
    if os.path.exists(vector_store_filename) and os.path.exists(extracted_data_filename):
        # Load FAISS index
        faiss_index: faiss.IndexFlatL2 = faiss.read_index(vector_store_filename)
        
        # Load rest of the data
        raw_data: pd.DataFrame = pd.read_parquet(extracted_data_filename)
        print("FAISS Index and extracted data loaded successfully.")
    
    else:
        print(f"Did not find file {extracted_data_filename}/{vector_store_filename}.\nProcessing your documents in data folder using processDocuments function")
        # Import utils file
        from .data_extractor import processDocuments 
        processDocuments()
        
        # Trying to load data once more
        faiss_index: faiss.IndexFlatL2 = faiss.read_index(vector_store_filename)
        
        # Load rest of the data
        raw_data: pd.DataFrame = pd.read_parquet(extracted_data_filename)
        print("FAISS Index and extracted data loaded successfully.")
        
    return ExtractedData(index=faiss_index,raw_data=raw_data)
    
def retrieveContext(question: str,
                    extracted_data: ExtractedData, 
                    similarity_threshold: int = 3) -> str:
    """
    Retrieve relevant context based on the input question

    Parameters
    ----------
    question : str
        input question from user
    extracted_data : ExtractedData
        vector_store and extracted data
    similarity_threshold : int, optional
        threshold for max number of similar options, by default 3

    Returns
    -------
    str
        relevant context in structured format
    """
    # Extracted data
    loaded_index: faiss.IndexFlatL2 = extracted_data.index
    loaded_data: pd.DataFrame = extracted_data.raw_data
    
    # Make sure that threshold is within possible limit
    number_of_indices: int = loaded_index.ntotal
    if number_of_indices < similarity_threshold:
        similarity_threshold = number_of_indices
    del number_of_indices # free unused memory
    
    # Embed the input question
    embedding_model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding_list: list[float] = embedding_model.embed_query(question)
    query_embedding: np.ndarray = np.array(query_embedding_list).reshape(1, -1)
    
    # Perform the search
    _distances: np.ndarray  # declare
    indices: np.ndarray  
    _distances, indices = loaded_index.search(query_embedding, k=similarity_threshold) # type:ignore
    indices = indices.flatten()
    
    # Extract relevant data
    relevant_context_data: pd.DataFrame = pd.DataFrame() # Initialize to avoid Unbound Error
    try:
       relevant_context_data = loaded_data.loc[indices]
    except Exception as e:
        if "[-1]" in str(e):
            return "No Relevant context was found" # Maybe add web search here later

    # Relevant context string in a structured format
    relevant_context_objects: pd.Series[str] = relevant_context_data.apply(
    lambda row: f"Relevant context was found in file: {row['metadata']['file']}, page: {row['metadata']['page']}\n\n{row['text']}\n" + 
                 (f"\nTable metnioned in above summary:\n\n{row['metadata']['original_content']}\n" if row['metadata']['type'] == "table" else ""),
    axis=1)
    
    relevant_context: str = "\n".join(relevant_context_objects) # Above line creates a pd.Series object convert it into string
    
    return relevant_context

def rag(question: str,
        extracted_data: ExtractedData,
        llm: Chatbot,
        similarity_threshold: int = 3) -> LLMResponse:
    """
    RAG operation, takes in a question and reponds to it using extracted context

    Parameters
    ----------
    question : str
        input question from user
    extracted_data : ExtractedData
        vector_store and extracted data
    llm: str
        Ollama model name
    similarity_threshold : int, optional
        threshold for max number of similar options, by default 3

    Returns
    -------
    LLMResponse
        response,context
    """
    # Prompt to the llm
    user_prompt: str = getUserPrompt(key="RAG")
    
    # Retrieve context and generate response
    context: str = retrieveContext(question,extracted_data,similarity_threshold)
    if context == "No Relevant context was found":
        return LLMResponse(response="Sorry, I couldn't find any relevant information regarding this topic.\
                           \nPlease reframe your question and try again.",
                           context=context)
    
    response: str =  llm.chat(user_prompt=user_prompt.format(context=context, question=question))    
    
    return LLMResponse(response=response, context=context)
    
if __name__ == "__main__":
    pass