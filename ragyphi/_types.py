#################################################
# Author        : Philip Varghese Modayil
# Last Update   : 11-03-2025
# Topic         : Custom Data Types 
#################################################

#####################################################################################
#                                     Imports
#####################################################################################
from typing import TypedDict
from typing_extensions import NotRequired


class OllamaMessage(TypedDict):
    role: str
    content: str
    images: NotRequired[list[bytes]] # Optional element