#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 05-03-2025
# Topic         : Chatbot with Ollama model
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
# LLM
import ollama
from ._types import OllamaMessage

# Typing
from typing import Union

# Prompts
from .prompt_template import getSystemPrompt

############################
# Custom Exception
############################
class ModelNotFoundError(Exception):
    """Custom exception to handle cases where the specified model cannot be found."""
    def __init__(self, model_name: str) -> None:
        super().__init__(f"Model '{model_name}' not found. Please pull it first using 'ollama pull {model_name}'.")
class Chatbot:
    def __init__(self,local_llm: str, system_prompt: str = getSystemPrompt(key="QA")) -> None:
        # Test if the model exists
        available_models: list[ollama._types.ListResponse.Model] = ollama.list()["models"] #type:ignore
        
        if not any(model["model"] == local_llm for model in available_models):
            raise ModelNotFoundError(local_llm)
        
        # LLM model name
        self.llm_model: str = local_llm
        
        # List to store the context
        self.chat_history: list[OllamaMessage] = [{'role': 'system', 'content': system_prompt}]
        
        # Context length
        self.context_length: int = 100
        
    def _addChatHistory(self, message: OllamaMessage) -> None:
        if len(self.chat_history) > self.context_length:
            print("Advised to start a fresh chat as the converation has been going on for long...\n")
            _: OllamaMessage = self.chat_history.pop(1) # Remove from beginning but keep the original system prompt
            
        self.chat_history.append(message)
       
    def chat(self, 
             user_prompt: str) -> str:
        
        # Add the input question to chat history
        self._addChatHistory({'role': 'user', 'content': user_prompt})
        
        
        response: Union[str,None] = ollama.chat(model=self.llm_model, messages=self.chat_history).message.content
        
        if response is None:
            return "No response was generated" # Make sure that a string is returned always
        
        # Add the model response to chat history
        self._addChatHistory({'role': 'assistant', 'content': response})
        
        return response
        