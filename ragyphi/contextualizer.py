#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 28.02.2025
# Topic         : Contextualizing using LLM
#################################################

# LLM
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# Image formatting
import base64
from io import BytesIO
from PIL import Image

############################
# Custom Exception
############################
class ModelNotFoundError(Exception):
    """Custom exception to handle cases where the specified model cannot be found."""
    def __init__(self, model_name):
        super().__init__(f"Model '{model_name}' not found. Please pull it first using 'ollama pull {model_name}'.")
        
#####################################################################################
#                                    Classes
#####################################################################################

class LMContextualizer:
    def __init__(self,local_llm: str) -> None:
        try:
            self.llm: ChatOllama = ChatOllama(model=local_llm, temperature=0)
        except Exception as e:
            if "404" in str(e):
                raise ModelNotFoundError(local_llm)
            else: 
                print(f"Unexpected error occured: {e}")
                import sys
                sys.exit(1) # exit the program
                
    def contextualizeDataWithLM(self,content_to_summarize: str) -> str:
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
        
        return self.llm.invoke(
        [SystemMessage(content=contextualizerInstruction)]
        + [HumanMessage(content=contextualizerPrompt.format(content_to_summarize=content_to_summarize))]
        ).content
        
class VLMContextualizer:
    def __init__(self,local_llm: str) -> None:
        try:
            self.llm: ChatOllama = ChatOllama(model=local_llm, temperature=0)
        except Exception as e:
            if "404" in str(e):
                raise ModelNotFoundError(local_llm)
            else: 
                print(f"Unexpected error occured: {e}")
                import sys
                sys.exit(1) # exit the program
    
    def convertImageToBase64(self,pil_image: Image.Image) -> str:
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str
                
    def contextualizeDataWithVLM(self,content_to_summarize: str) -> str:
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
        
        return self.llm.invoke(
        [SystemMessage(content=contextualizerInstruction)]
        + [HumanMessage(content=contextualizerPrompt.format(content_to_summarize=content_to_summarize))]
        ).content
