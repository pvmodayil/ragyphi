# ragyphi
An entire RAG (Retrieval-Augmented Generation) pipeline library designed to streamline the integration of language models with retrieval systems.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation

To get started with `ragyphi`, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pvmodayil/ragyphi.git
   cd ragyphi
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   conda env create -f environment.yml  # Create an environment named 'RAG'
   conda activate RAG  # Activate the newly created environment
   ```
3. **Install Ollama**
   : Make sure you have Ollama installed on your system. Then, run the following command in your terminal
   ```bash
   ollama pull llama3.2:3b-instruct-fp16
   ```
## Usage
Create a directory 'data' with the pdf files for RAG usage and run the following code.
```python
  # Import necessary libraries
  import ragyphi as rp
  from langchain_community.chat_models import ChatOllama
  
  # Initialize the local language model (LLM)
  local_llm = "llama3.2:3b-instruct-fp16"
  llm = ChatOllama(model=local_llm, temperature=0)
  
  # Load data for processing
  data = rp.loadData()
  
  # Define your question and retrieve response using RAG pipeline
  question = "your-question-here"
  result = rp.rag(question, data, llm)
  
  # Print the result and context for review
  print("Response:", result.response)
  print("Context:", result.context)
```
