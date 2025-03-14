#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 14-03-2025
# Topic         : LLM Prompt Templates
#################################################

# system definition
###################################################################
SYSTEM_PROMPT: dict[str,str] = {
    # System prompt for scientific research analysis
    "SCIENTIFIC_ANALYST": """You are an expert assistant working along with researchers and engineers in the {domain} domain. 
    You are capable of analyzing scientific documents like data sheets, research papers, reports, thesis, articles, reviews, and presentations.""",
    
    "SCIENTIFIC_ASSISTANT": """You are an expert assistant working along with researchers and engineers in the {domain} domain.
    You are capable of analyzing a given contextual information and reponding to questions accurately.
    You will say that you don't know the answer if you don't know the answer, and ask clarification questions if necessary.""",
    
    # General question answer prompt
    "QA": """You are a helpful assistant capable of answering questions with accuracy in a concise format.
    You will say that you don't know the answer if you don't know the answer, and ask clarification questions if necessary.""",
}

def getSystemPrompt(key: str, **kwargs) -> str:
    if key not in SYSTEM_PROMPT:
        raise KeyError(f"Prompt key '{key}' not found in SYSTEM_PROMPT dictionary.")
    
    prompt: str = SYSTEM_PROMPT[key]
    
    # Check if the prompt contains placeholders and format it if necessary
    if kwargs:
        try:
            formatted_prompt: str = prompt.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing key '{e.args[0]}' in provided parameters for prompt key '{key}'.")
        return formatted_prompt
    
    return prompt

# user prompts
####################################################################
USER_PROMPT: dict[str,str]= {
    "RAG": """Here is the context to use to answer the question:

    {context} 
    
    Think carefully about the above context and review the user question given below:

    {question}

    Based on the context answer the question accurately and structure your response in the following format:
    Answer: Provide a concise answer to the question.
    Context: Explain the relevant context from the provided information.
    Questions: Suggest follow-up questions for clarification or further exploration."""
}

def getUserPrompt(key: str) -> str:
    return USER_PROMPT[key]

# Contextualization
####################################################################
CONTEXTUALIZER_INSTRUCTION: str = """You are a helpful assistant capable of summarizing {content_type} for retrieval."""
CONTEXTUALIZER_PROMPT: dict[str,str] = {
    "LM": """{instruction}

        Carefully analyse the {content_type} data from the document and provide a detailed summary.
        These summaries will be embedded and used to retrieve the raw {content_type} elements.
        Also generate hypothetical questions that can be answered based on the given context.

        Document to be summarized:
        {content_to_summarize}

        Please structure your response in the following format:
        1. A concise summary of the table or text that is well optimized for retrieval.
        2. List the key observations and relevant metrics.
        3. List of the major keywords.
        4. A list of exactly 3 hypothetical questions that the above document could be used to answer.
        """,
    "VLM": """{instruction}

        Carefully analyze the provided text and/or image data and provide a detailed summary.
        These summaries will be used for retrieval purposes.
        Also generate hypothetical questions that can be answered based on the given context.
        
        Use this given text for additional information regarding the image:
        {content_to_summarize}

        Please structure your response in the following format:
        1. A concise description of the image that is well optimized for retrieval.
        2. List the key observations and relevant details.
        3. List of the major keywords or visual elements.
        4. A list of exactly 3 hypothetical questions that the provided content could be used to answer.
        """
}
def getContextualizerPrompt(key: str,
                            content_to_summarize: str,
                            content_type: str) -> str:
    
    
    instruction: str = CONTEXTUALIZER_INSTRUCTION.format(content_type=content_type)
    prompt: str = CONTEXTUALIZER_PROMPT[key]
    return prompt.format(instruction=instruction,
                            content_to_summarize=content_to_summarize) 