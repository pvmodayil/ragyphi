# ragyphi
An entire RAG (Retrieval-Augmented Generation) pipeline library designed to streamline the integration of language models with retrieval systems.
ragyphi can extract text, table, and image data from your pdf files and uses clever llm-based summarization techniques to augment the extracted data for efficient retrieval. 

## Why Choose ragyphi?

In the realm of Retrieval-Augmented Generation (RAG) pipelines, many frameworks struggle to effectively comprehend documents and perform accurate extractions. Enter **ragyphi**—a solution designed with precision in mind.

### Key Features

- **Effortless PDF Extraction**: Leveraging the power of **pdfplumber**, a lightweight PDF extraction library, ragyphi enables seamless text and table extraction without the need for cumbersome system-level installations.
  
- **Enhanced Image Detection**: Current libraries often fall short when it comes to image extraction, resulting in lost data and subpar user query responses. ragyphi tackles this challenge head-on by employing a fine-tuned **YOLOv11 model**. This model expertly detects images within PDF pages, cropping them accurately based on predicted bounding box values.

- **Smart Data Augmentation**: Once extracted, the data is not only summarized but also enriched with hypothetical questions. This approach enhances semantic similarity searches between user queries and documents, ensuring more precise results.

- **Local LLM Inferences**: With the integration of **Ollama models**, all large language model (LLM) inferences occur locally on your machine, prioritizing your data privacy while delivering powerful insights.

### Who Can Benefit?

Ragyphi is tailored for researchers and professionals who rely heavily on sensitive data sheets and values in their scientific work. However, its versatility means that anyone looking to create a local database with LLM assistance can benefit from what ragyphi has to offer!

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
   pip install uv # Install uv
   uv --version # Verify uv installation

   uv venv --python 3.12 # Create venv

   source .venv/bin/activate  # On Unix-based systems
   # or
   .venv\Scripts\activate  # On Windows

   uv pip install -r pyproject.toml # Install dependencies
   ```
3. **Install Ollama**
   : Make sure you have Ollama >= 0.5.13 installed on your system [Ollama download](https://ollama.com/download). Then, run the following command in your terminal
   ```bash
   ollama pull llama3.2:3b-instruct-fp16
   ollama pull granite3.2-vision
   ```
## Usage
Create a directory 'data' with the pdf files for RAG usage and run the following code in a new python file.
```python
  # Import necessary libraries
  from ragyphi import ragyphi as rp
  from ragyphi.ollama_chat import Chatbot
  from ragyphi.prompt_template import getSystemPrompt
  
  # Load data for processing
  data: rp.ExtractedData = rp.loadData()

  # Initialize the local language model (LLM)
  local_llm: str = "llama3.2:3b-instruct-fp16" # If you like to use other models pull them with ollama pull first
  system_prompt: str = getSystemPrompt(key="SCIENTIFIC_ASSISTANT", domain="Physics")
  llm: Chatbot = Chatbot(local_llm=local_llm, system_prompt=system_prompt)
  
  # Define your question and retrieve response using RAG pipeline
  question: str = "your-question-here"
  result: rp.LLMResponse = rp.rag(question, data, llm)
  
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
