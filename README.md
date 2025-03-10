# ragyphi
An entire RAG (Retrieval-Augmented Generation) pipeline library designed to streamline the integration of language models with retrieval systems.
ragyphi can extract text and table data from your pdf files and uses clever llm-based summarization techniques to augment the extracted data for efficient retrieval. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Storage](#storage)

## Installation

To get started with `ragyphi`, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pvmodayil/ragyphi.git
   cd ragyphi
   ```
2. **Create a virtual environment with uv:**
   ```bash
   uv sync  # Will create the virtual environment and install the required dependecies
   source .venv/bin/activate  # On Unix-based systems
   # or
   .venv\Scripts\activate  # On Windows
   ```
3. **Install Ollama**
   : Make sure you have Ollama >= 0.5.13 installed on your system [Ollama download](https://ollama.com/download). Then, run the following command in your terminal
   ```bash
   ollama pull llama3.2:3b-instruct-fp16
   ollama pull granite3.2-vision
   ```
## Usage
Create a directory 'data' with the pdf files for RAG usage and run the following code.
```python
  # Import necessary libraries
  from ragyphi import ragyphi as rp
  from ragyphi.ollama_chat import Chatbot
  
  # Initialize the local language model (LLM)
  local_llm = "llama3.2:3b-instruct-fp16" # If you like to use other models pull them with ollama pull first
  system_prompt = "You are a helpful assistant capable of answering scientific questions."
  llm = Chatbot(local_llm=local_llm, system_prompt=system_prompt)
  
  # Load data for processing
  data = rp.loadData()
  
  # Define your question and retrieve response using RAG pipeline
  question = "your-question-here"
  result = rp.rag(question, data, llm)
  
  # Print the result and context for review
  print("Response:", result.response)
  print("Context:", result.context)
```
## Storage
Each extracted data unit is stored in the following format. This list is then converted into a datframe and stored as parquet files.
```json
{
    "uuid": "unique id",
    "text": "generated summary",
    "metadata": {
        "file": "name of the document",
        "page": "page number", 
        "type": "type of the extracted data",
        "original_content": "original content which was summarized"
    }
}
