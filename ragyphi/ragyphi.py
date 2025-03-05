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
import ollama
# from ollama._types import ResponseError

############################
# Custom Exception
############################
class ModelNotFoundError(Exception):
    """Custom exception to handle cases where the specified model cannot be found."""
    def __init__(self, model_name):
        super().__init__(f"Model '{model_name}' not found. Please pull it first using 'ollama pull {model_name}'.")


############################
# Custom DataClasses
############################
@dataclass(frozen=True)
class ExtractedData:
    index: faiss.swigfaiss_avx2.IndexFlatL2
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
    query_embedding: list = embedding_model.embed_query(question)
    query_embedding: np.ndarray = np.array(query_embedding).reshape(1, -1)
    
    # Perform the search
    distances, indices = loaded_index.search(query_embedding, similarity_threshold)
    indices = indices.flatten()
    
    # Extract relevant data
    try:
       relevant_context_data: pd.DataFrame = loaded_data.loc[indices]
    except Exception as e:
        if "[-1]" in str(e):
            return "No Relevant context was found" # maybe add web search here later

    # Relevant context string in a structured format
    relevant_context_objects: list[str] = relevant_context_data.apply(
    lambda row: f"Relevant context was found in file: {row['metadata']['file']}, page: {row['metadata']['page']}\n\n{row['text']}\n" + 
                 (f"\nTable metnioned in above summary:\n\n{row['metadata']['original_content']}\n" if row['metadata']['type'] == "table" else ""),
    axis=1)
    
    relevant_context: str = "\n".join(relevant_context_objects) # above line creates a pd.Series object convert it into string
    
    return relevant_context

def rag(question: str,
        extracted_data: ExtractedData,
        llm: str,
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
    system_prompt: str = """You are a PCB designer assistant capable of answering questions \
        based on relevant scientific information."""
     
    user_prompt: str = """
    Here is the context to use to answer the question:\

    {context} 


    Think carefully about the above context which may include tabular data. \

    Now, carefully review the user question given below and provide a clear answer:\

    {question}\

    Please structure your response in the following format:
    Answer: Provide a concise answer to the question.
    Context: Explain the relevant context from the provided information.
    Questions: Suggest follow-up questions for clarification or further exploration.

    If you don't know the answer, just say that you don't know, don't try to make up an answer."""
    
    # Retrieve context and generate response
    context: str = retrieveContext(question,extracted_data,similarity_threshold)
    if context == "No Relevant context was found":
        return LLMResponse(response="Sorry, I couldn't find any relevant information regarding this topic.\
                           \nPlease reframe your question and try again.",
                           context=context)
        
    response = ollama.chat(model=llm, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt.format(context=context, question=question)}
        ])
        
    return response['message']['content']
    
if __name__ == "__main__":
    pass