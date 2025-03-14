#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 11.03.2025
# Topic         : Contextualizing using LLM
#################################################

# LLM
import ollama

# Image formatting
from io import BytesIO
from PIL import Image

# Typing
from ._types import OllamaMessage

# Prompts
from .prompt_template import getContextualizerPrompt, getSystemPrompt
    
############################
# Custom Exception
############################
class ModelNotFoundError(Exception):
    """Custom exception to handle cases where the specified model cannot be found."""
    def __init__(self, model_name: str) -> None:
        super().__init__(f"Model '{model_name}' not found. Please pull it first using 'ollama pull {model_name}'.")
        
#####################################################################################
#                                    Classes
#####################################################################################
class LMContextualizer:
    def __init__(self, 
                 domain: str, 
                 local_llm: str) -> None:
        # Test if the model exists
        available_models: list[dict[str,str]] = ollama.list()["models"]
        
        if not any(model["model"] == local_llm for model in available_models):
            raise ModelNotFoundError(local_llm)
        
        # If exists initialize class
        self.model: str = local_llm
        self.system_prompt: str = getSystemPrompt(key="SCIENTIFIC_ANALYST", domain=domain)

    def contextualizeDataWithLM(self, 
                                content_type: str,
                                content_to_summarize: str) -> str:
        """
        Takes in text to be summarised and summarises it.
        Parameters
        ----------
        content_type : str
            type of the content
        content_to_summarize : str
            text / table along with context that needs to be summarised 

        Returns
        -------
        str
            summarised text 
        """
        # Contextualize prompt
        ###############################################
        contextualizerPrompt: str = getContextualizerPrompt(key="LM",
                                                            content_to_summarize=content_to_summarize,
                                                            content_type=content_type,)
        
        response: ollama.ChatResponse = ollama.chat(model=self.model, messages=[
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': contextualizerPrompt}
        ])
        
        return response['message']['content']

class VLMContextualizer:
    def __init__(self, domain: str,
                 local_vllm: str) -> None:
        # Test if the model exists
        available_models: list[dict[str,str]] = ollama.list()["models"]
        
        if not any(model["model"] == local_vllm for model in available_models):
            raise ModelNotFoundError(local_vllm)
        
        # If exists initialize class
        self.model: str = local_vllm
        self.system_prompt: str = getSystemPrompt(key="SCIENTIFIC_ANALYST", domain=domain)
        
    @staticmethod
    def _get_image_bytes(image: Image.Image) -> bytes:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    def contextualizeDataWithLM(self, additional_text: str, image: Image.Image) -> str:
        # Contextualize prompt
        ###############################################
        contextualizerPrompt: str = getContextualizerPrompt(key="VLM",
                                                            content_to_summarize=additional_text,
                                                            content_type="image",)
        image_data: bytes = self._get_image_bytes(image)
        
        messages: list[OllamaMessage] = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': contextualizerPrompt,'images': [image_data]},
            ]

        response: ollama.ChatResponse = ollama.chat(model=self.model, messages=messages)
        
        return response['message']['content']       