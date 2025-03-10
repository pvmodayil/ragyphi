#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 28.02.2025
# Topic         : Contextualizing using LLM
#################################################

# LLM
import ollama

# Image formatting
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
    def __init__(self, local_llm: str) -> None:
        # Test if the model exists
        available_models = ollama.list()["models"]
        
        if not any(model["model"] == local_llm for model in available_models):
            raise ModelNotFoundError(local_llm)
        
        # If exists initialize class
        self.model = local_llm

    def contextualizeDataWithLM(self, content_to_summarize: str) -> str:
        """
        Takes in text to be summarised and summarises it.
        Parameters
        ----------
        content_to_summarize : str
            text / table along with context that needs to be summarised 

        Returns
        -------
        str
            summarised text 
        """
        # Contextualize prompt
        ###############################################
        contextualizerInstruction = """You are a helpful assistant capable of summarizing texts and tables for retrieval."""
        
        contextualizerPrompt = f"""{contextualizerInstruction}

        Carefully analyse the text or table data from the document and provide a detailed summary.
        These summaries will be embedded and used to retrieve the raw text or table elements.
        Also generate hypothetical questions that can be answered based on the given context.

        Document to be summarized:
        {content_to_summarize}

        Please structure your response in the following format:
        1. A concise summary of the table or text that is well optimized for retrieval.
        2. List the key observations and relevant metrics.
        3. List of the major keywords.
        4. A list of exactly 3 hypothetical questions that the above document could be used to answer.
        """
        
        response = ollama.chat(model=self.model, messages=[
            {'role': 'system', 'content': contextualizerInstruction},
            {'role': 'user', 'content': contextualizerPrompt}
        ])
        
        return response['message']['content']

class VLMContextualizer:
    def __init__(self, local_llm: str = "granite3.2-vision:latest") -> None:
        # Test if the model exists
        available_models = ollama.list()["models"]
        
        if not any(model["model"] == local_llm for model in available_models):
            raise ModelNotFoundError(local_llm)
        
        # If exists initialize class
        self.model = local_llm

    def _get_image_bytes(self, image: Image.Image) -> bytes:
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

    def contextualizeDataWithLM(self, additional_text: str, image: Image.Image = None) -> str:
        contextualizerInstruction = """You are a helpful assistant capable of analyzing and summarizing both text and images."""
        
        contextualizerPrompt = f"""{contextualizerInstruction}

        Carefully analyze the provided text and/or image data and provide a detailed summary.
        These summaries will be used for retrieval purposes.
        Also generate hypothetical questions that can be answered based on the given context.

        {"Image data is provided." if image else ""}
        
        Use this given text for additional information regarding the image:
        {additional_text}

        Please structure your response in the following format:
        1. A concise description of the image that is well optimized for retrieval.
        2. List the key observations and relevant details.
        3. List of the major keywords or visual elements.
        4. A list of exactly 3 hypothetical questions that the provided content could be used to answer.
        """
        
        messages = [
            {'role': 'system', 'content': contextualizerInstruction},
            {'role': 'user', 'content': contextualizerPrompt}
        ]

        if image:
            image_data = self._get_image_bytes(image)
            messages.append({
                'role': 'user',
                'content': "Please analyze this image along with the provided text.",
                'images': [image_data]
            })

        response = ollama.chat(model=self.model, messages=messages)
        
        return response['message']['content']       