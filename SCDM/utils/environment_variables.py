# importing os module for environment variables
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv

import torch

__all__ = [
    "TORCH_DEVICE", 
    "HP_TOKEN"
]

# torch
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# loading variables from .env file
load_dotenv()

# getenv
HP_TOKEN = os.getenv("HP_TOKEN")
