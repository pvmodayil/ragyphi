#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 14.02.2025
# Topic         : Document pre process
# Documents     : pdf
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
"""
To Do: 
    Add multi document processing capabilities
"""

#####################################################################################
#                                     Imports
#####################################################################################
import os
import uuid
from tqdm import tqdm

# pdf
import pdfplumber

import pandas as pd
import numpy as np

# Database
import faiss

# Text extraction
from langchain_huggingface import HuggingFaceEmbeddings

# LLM
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Typing
from typing import List
from dataclasses import dataclass

@dataclass
class ExtractedData:
    index: faiss.swigfaiss_avx2.IndexFlatL2
    raw_data: pd.DataFrame

@dataclass
class LLMResponse:
    response: str
    context: str

#####################################################################################
#                                    Functions
#####################################################################################
# Create the directories
def create_directories(base_dir):
    directories = ["images", "text","tables"]
    for dir in directories:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)
        
###########################################
#        Document Processing: PDF
###########################################   
#####################################################################################
#                       Extracted Data Storage Schematics
#####################################################################################
# {
# "uuid": unique id,
# "text": generated summary,
# "metadata":{
#     "file": name of the document,
#     "page": page number, 
#     "type": type of the extarcted data,
#     "original_content": original content which was summarized
#      }
# } 
def extractAndContextualizePDFPage(base_dir: str, 
                      pdf_path: str, 
                      page: pdfplumber.page.Page,  
                      extracted_items: list,
                      llm: ChatOllama) -> List[dict]:
    
    # Contextualize prompt
    ###############################################
    contextualizerInstruction = """You are a helpful assistant capable of summarizing texts and tables for retrieval."""
    
    contextualizerPrompt = """ Carefully analyse the text or table data from the document and provide a detailed summary.\
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
    # Get unique identifiers for the page
    ###########################################
    page_number = page.page_number
    pdf_filename = os.path.basename(pdf_path)[:-4]
    
    # Extract text and contextualize
    ###########################################
    page_content_text = page.extract_text()
    
    # Summarize
    content_to_summarize = f"Page text:\n{page_content_text}"
    generated_context_text = llm.invoke(
    [SystemMessage(content=contextualizerInstruction)]
    + [HumanMessage(content=contextualizerPrompt.format(content_to_summarize=content_to_summarize))]
    ).content  
    
    # Store in structured format
    extracted_items.append({
                "uuid": str(uuid.uuid4()),
                "text": generated_context_text,
                "metadata":{
                    "file": pdf_filename,
                    "page": page_number, 
                    "type": "text",
                    "original_content": page_content_text}
                })
    
    # Save text
    text_filename = os.path.join(base_dir,"text",f"{pdf_filename}_{page_number}_text.txt") 
    text_summary_filename = os.path.join(base_dir,"text",f"{pdf_filename}_{page_number}_context.md") 
    try:
        with open(text_filename, 'w') as f:
            f.write(page_content_text)
        with open(text_summary_filename, 'w') as f:
            f.write(generated_context_text)
    except OSError:
        print(f"Something wrong with file name: {text_filename}")
    
    # Extract tables
    #########################################
    page_content_tables = page.extract_tables()
    for table_id,table in enumerate(page_content_tables):
        # Convert extracted table to pandas dataframe
        df = pd.DataFrame(table[1:], columns=table[0])  # Create DataFrame
        table_content_text = df.to_markdown()
        
        # Contextualize the table using llm
        content_to_summarize = f"Page text:\n{page_content_text}\nTable:\n{table_content_text}"
        generated_context_table = llm.invoke(
        [SystemMessage(content=contextualizerInstruction)]
        + [HumanMessage(content=contextualizerPrompt.format(content_to_summarize=content_to_summarize))]
        ).content  
        
        extracted_items.append({
                    "uuid": str(uuid.uuid4()), 
                    "text": generated_context_table,
                    "metadata":{
                        "file": pdf_filename,
                        "page": page_number,
                        "type": "table",
                        "original_content": table_content_text}
                    })

        # Save table
        table_filename = os.path.join(base_dir,"tables",f"{pdf_filename}_{page_number}_table_{table_id}.csv") 
        table_summary_filename = os.path.join(base_dir,"tables",f"{pdf_filename}_{page_number}_context_{table_id}.md") 
        try:
            df.to_csv(table_filename,index=False)
            with open(table_summary_filename, 'w') as f:
                f.write(generated_context_table)
        except OSError:
             print(f"Something wrong with file name: {table_filename}") 
    
    return extracted_items

def loadPDFAndExtract(base_dir: str, 
                      extracted_items: list, 
                      llm: ChatOllama) -> List[dict]:
    
    for file in os.listdir(base_dir):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(base_dir, file)
            
            with pdfplumber.open(pdf_path) as pdf:
                for page in tqdm(pdf.pages, desc=f"Processing {pdf_path} pages"):
                    # Extract context
                    extracted_items = extractAndContextualizePDFPage(base_dir, pdf_path, page, extracted_items, llm)
            
            print(f"Texts and tables with context are extracted from file {pdf_path}")       
        
    return extracted_items   

def convertExtractedItemsToDF(extracted_items: List[dict]) -> pd.DataFrame:
    # Convert the list of dictionaries into a single dataframe
    return pd.DataFrame(extracted_items)

###########################################
#        Embed and Store Data
###########################################  
def embedExractedData(extracted_data_df: pd.DataFrame) -> pd.DataFrame:
    # Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        # The extracted_data_df has column "text" which contains the text chunks to be embedded
        print("Embedding text chunks..........")
        extracted_data_df['embedding'] = extracted_data_df["text"].apply(lambda x: embedding_model.embed_query(x))
    except KeyError:
        raise Exception("Did not find column 'text' in the extracted_data dataframe.\nMake sure extractable text is present in the document.")
    
    return extracted_data_df

def indexAndStore(extracted_data_df: pd.DataFrame) -> None:
    print("Storing vectors and metadata..........")
    try:
        # Convert the embeddings(array) to an array of arrays
        all_embeddings = np.array(extracted_data_df['embedding'].apply(np.array).tolist()).astype('float32')
    except KeyError:
        raise Exception("Did not find the column 'embedding' in the extracted_data dataframe.")
    
    # Define the dimension of your embedding vectors
    embedding_vector_dimension = all_embeddings.shape[1]

    # Create FAISS Index
    index = faiss.IndexFlatL2(embedding_vector_dimension)

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
    extracted_data_filename = os.path.join(os.getcwd(),"vector_store",'extracted_data.parquet')
    extracted_data_df.to_parquet(extracted_data_filename, compression='snappy')

def processDocuments(local_llm: str = "llama3.2:3b-instruct-fp16") -> None:
    
    print("Creating directory data.......\n Please include any documents that need to be extracted if they are not already provided.")
    # Add more supported document type later
    print("Supported document types:\n- PDF")
    
    # Create data directory
    ###########################################
    os.makedirs("data", exist_ok=True)
    base_dir = os.path.join(os.getcwd(),"data")
    create_directories(base_dir)
    
    # Load llm
    ###########################################
    llm = ChatOllama(model=local_llm, temperature=0)
    
    # Extract information from pdf files
    ###########################################
    extracted_items = []
    extracted_items = loadPDFAndExtract(base_dir,extracted_items,llm)
    
    # Convert to dataframe
    ###########################################
    extracted_data_df = convertExtractedItemsToDF(extracted_items)
    
    # Embed extracted text
    ###########################################
    extracted_data_df = embedExractedData(extracted_data_df)
    
    # Index with FAISS and store data
    indexAndStore(extracted_data_df)
    
###########################################
#        Load and Retrieve
###########################################      
def loadData() -> ExtractedData:
    vector_store_filename = os.path.join(os.getcwd(),"vector_store",'faiss_index.index')
    extracted_data_filename = os.path.join(os.getcwd(),"vector_store",'extracted_data.parquet')
    
    if os.path.exists(vector_store_filename) and os.path.exists(extracted_data_filename):
        # Load FAISS index
        faiss_index = faiss.read_index(vector_store_filename)
        
        # Load rest of the data
        raw_data = pd.read_parquet(extracted_data_filename)
        print("FAISS Index and extracted data loaded successfully.")
    
    else:
        print(f"Did not find file {extracted_data_filename}/{vector_store_filename}.\nProcessing your documents in data folder using processDocuments function")
        processDocuments()
        
        # Trying to load data once more
        faiss_index = faiss.read_index(vector_store_filename)
        
        # Load rest of the data
        raw_data = pd.read_parquet(extracted_data_filename)
        print("FAISS Index and extracted data loaded successfully.")
        
    return ExtractedData(index=faiss_index,raw_data=raw_data)
    
def retrieveContext(question: str,
                    extracted_data: ExtractedData, 
                    similarity_threshold: int = 3) -> str:
    
    # Extracted data
    loaded_index = extracted_data.index
    loaded_data = extracted_data.raw_data

    # Embed the input question
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embedding_model.embed_query(question)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Perform the search
    distances, indices = loaded_index.search(query_embedding, similarity_threshold)
    indices = indices.flatten()

    # Extract relevant data
    relevant_context_data = loaded_data.loc[indices]
    

    # Relevant context string in a structured format
    relevant_context_object = relevant_context_data.apply(
    lambda row: f"Relevant context was found in file: {row['metadata']['file']}, page: {row['metadata']['page']}\n\n{row['text']}\n" + 
                 (f"\nTable metnioned in above summary:\n\n{row['metadata']['original_content']}\n" if row['metadata']['type'] == "table" else ""),
    axis=1)
    
    relevant_context = "\n".join(relevant_context_object) # above line creates a pd.Series object convert it into string
    
    return relevant_context

def rag(question: str,
        extracted_data: ExtractedData,
        llm: ChatOllama,
        similarity_threshold: int = 3) -> LLMResponse:
    # Prompt to the llm
    rag_prompt = """You are a PCB designer assistant for providing required information. 

    Here is the context to use to answer the question:

    {context} 


    Think carefully about the above context which may include tabular data. 

    Now, carefully review the user question given below and provide a clear answer:

    {question}

    Please structure your response in the following format:
    Answer: Provide a concise answer to the question.
    Context: Explain the relevant context from the provided information.
    Questions: Suggest follow-up questions for clarification or further exploration.

    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Response:"""
    
    # Retrieve context and generate response
    context = retrieveContext(question,extracted_data,similarity_threshold)
    
    rag_prompt_formatted = rag_prompt.format(context=context, question=question)
    response = llm.invoke([HumanMessage(content=rag_prompt_formatted)]).content

    return LLMResponse(response=response,context=context)
    
#####################################################################################
#                                   MAIN Functions
#####################################################################################

    

if __name__ == "__main__":
    pass