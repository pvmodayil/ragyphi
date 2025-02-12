#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 06.02.2025
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
from tqdm import tqdm

# pdf
import pdfplumber

import pandas as pd
import numpy as np

# Database
import faiss

# Text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    metadata: pd.DataFrame

@dataclass
class LLMResponse:
    response: str
    context: str

#####################################################################################
#                                    Functions
#####################################################################################
# Create the directories
def create_directories(base_dir):
    directories = ["images", "text", "context","tables", "page_images"]
    for dir in directories:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)
        
###########################################
#        Document Processing: PDF
###########################################    
def extractAndContextualizePDFPage(base_dir: str, 
                      pdf_path: str, 
                      page: pdfplumber.page.Page,  
                      extracted_items: list,
                      llm: ChatOllama) -> List[dict]:
    
    # Contextualize prompt
    ###############################################
    contextualizerInstruction = """You are a helpful assistant capable of describing tabular data."""
    
    contextualizerPrompt = """ Given the following table and its context from the original document,
    provide a detailed description of the table. Include the table heading, context of the table, important values and information from the table.

    Original Document Context:
    {document_context}

    Table Content:
    {table_content}

    Please provide:
    1. A comprehensive description of the table.
    2. Important values from the table
    """
    # Get page number
    ###########################################
    page_number = page.page_number
    pdf_filename = os.path.basename(pdf_path)[:-4]
    # Extract text
    ###########################################
    page_content_text = page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200)
    chunks = text_splitter.split_text(page_content_text)
    
    for id, chunk in enumerate(chunks):
        text_file_name = os.path.join(base_dir,"text",f"{pdf_filename}_text_{page_number}_{id}.txt")
        try:
            with open(text_file_name, 'w') as f:
                f.write(chunk)
        except OSError:
            print(f"Something wrong with file name: {text_file_name}")
            
        extracted_items.append({"page": page_number, 
                    "type": "text", 
                    "text": chunk, 
                    "path": pdf_filename})
    
    
    # Extract tables
    #########################################
    table_content_text = ""
    page_content_tables = page.extract_tables()
    for table_id,table in enumerate(page_content_tables):
        df = pd.DataFrame(table[1:], columns=table[0])  # Create DataFrame
        table_file_name = os.path.join(base_dir,"tables",f"{pdf_filename}_table_{page_number}_{table_id}.csv") 
        
        try:
            df.to_csv(table_file_name,index=False)
        except OSError:
             print(f"Something wrong with file name: {table_file_name}") 
        
        table_content_text += "\n\n" + df.to_markdown()
    
    # Contextualize using llm
    #########################################
    generated_context = llm.invoke(
    [SystemMessage(content=contextualizerInstruction)]
    + [HumanMessage(content=contextualizerPrompt.format(document_context=page_content_text,table_content=table_content_text))]
    )    
    generated_context_summary = "{summary}\n\n# Relevant Tables:\n{tables}".format(summary=generated_context.content, tables=table_content_text)
    
    # Save data
    try:
        
        context_file_name = os.path.join(base_dir,"context",f"{pdf_filename}_context_{page_number}.md")
        # Save the generated page context
        with open(context_file_name, 'w') as f:
                f.write(generated_context_summary)
        
    except OSError:
        print(f"Something wrong with file name: {context_file_name}")    
    
    extracted_items.append({"page": page_number, 
                            "type": "page_context", 
                            "text": generated_context_summary, 
                            "path": pdf_filename})    
    
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
    df_extracted_items = pd.DataFrame(extracted_items)

    df_txt = df_extracted_items[df_extracted_items["type"] == "text"]
    df_metadata = df_extracted_items[df_extracted_items["type"] != "text"]
    df_merged = pd.merge(df_txt, df_metadata, on=["page","path"], how="inner",suffixes=("", "_metadata"))

    return df_merged

###########################################
#        Embed and Store Data
###########################################  
def embedExractedData(extracted_data_df: pd.DataFrame) -> pd.DataFrame:
    # Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    try:
        # The extracted_data_df has column "text" and "text_metadata"
        # The column "text" has text chunk values which need to be embedded
        print("Embedding text chunks..........")
        extracted_data_df['embedding'] = extracted_data_df["text"].apply(lambda x: embedding_model.embed_query(x))
    except KeyError:
        raise Exception("Did not find column 'text' in the extracted_data dataframe.\nMake sure text is present in the document.")
    
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
        metadata = pd.read_parquet(extracted_data_filename)
        print("FAISS Index and extracted data loaded successfully.")
    
    else:
        print(f"Did not find file {extracted_data_filename}/{vector_store_filename}.\nProcessing your documents in data folder using processDocuments function")
        processDocuments()
        
        # Trying to load data once more
        faiss_index = faiss.read_index(vector_store_filename)
        
        # Load rest of the data
        metadata = pd.read_parquet(extracted_data_filename)
        print("FAISS Index and extracted data loaded successfully.")
        
    return ExtractedData(index=faiss_index,metadata=metadata)
    
def retrieveContext(question: str,
                    extracted_data: ExtractedData, 
                    similarity_threshold: int = 3) -> str:
    
    # Extracted data
    loaded_index = extracted_data.index
    loaded_data = extracted_data.metadata

    # Embed the input question
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embedding_model.embed_query(question)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Perform the search
    distances, indices = loaded_index.search(query_embedding, similarity_threshold)
    indices = indices.flatten()

    # Extract relevant data
    relevant_context_data = loaded_data.loc[indices]
    relevant_context_data["combined_path"] = relevant_context_data["path"] + "_" + relevant_context_data["page"].astype(str)
    unique_metadata = relevant_context_data.drop_duplicates(subset=["combined_path"])

    # Relevant text chunks - unstructured
    relevant_context_text = '\n\n'.join(relevant_context_data["text"])

    # Relevant metadata - structured summaries of pages
    metadata_strings = unique_metadata.apply(lambda row: f"{row['text_metadata']}\nAbove information was found in file: {row['path']}, page: {row['page']}", axis=1)
    relevant_context_metadata = '\n\n'.join(metadata_strings)
    
    # Fromatting context
    relevant_context = f"Unstructured Text Information:\n\n{relevant_context_text}\n\nStructured Page Summaries:\n\n{relevant_context_metadata}"
    
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