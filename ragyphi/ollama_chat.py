#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 05-03-2025
# Topic         : 
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
# LLM
import ollama

# Prompts
from ._prompt_template import getSystemPrompt

############################
# Custom Exception
############################
class ModelNotFoundError(Exception):
    """Custom exception to handle cases where the specified model cannot be found."""
    def __init__(self, model_name: str) -> None:
        super().__init__(f"Model '{model_name}' not found. Please pull it first using 'ollama pull {model_name}'.")
class Chatbot:
    def __init__(self,local_llm: str) -> None:
        # Test if the model exists
        available_models: list[ollama._types.ListResponse.Model] = ollama.list()["models"] #type:ignore
        
        if not any(model["model"] == local_llm for model in available_models):
            raise ModelNotFoundError(local_llm)
        
        # LLM model name
        self.llm_model: str = local_llm
        
        # List to store the context
        self.chat_history: list[dict[str,str]] = [{'role': 'system', 'content': getSystemPrompt(key="QA")}]
        
        # Context length
        self.context_length: int = 100
        
        # Default response
        self.default_response: str = "No response was generated" # Adding a type guard
        
    def addChatHistory(self, message: dict[str, str]) -> None:
        if len(self.chat_history) > self.context_length:
            print("Advised to start a fresh chat as the converation has been going on for long...\n")
            _: dict[str, str] = self.chat_history.pop(1) # Remove from beginning but keep the original system prompt
            
        self.chat_history.append(message)
       
    def chat(self, 
             user_prompt: str) -> str:
        
        # Add the input question to chat history
        self.addChatHistory({'role': 'user', 'content': user_prompt})
        
        
        response: str = ollama.chat(model=self.llm_model, messages=self.chat_history).message.content or \
            self.default_response
        
        # Add the model response to chat history
        self.addChatHistory({'role': 'assistant', 'content': response})
        
        return response
        