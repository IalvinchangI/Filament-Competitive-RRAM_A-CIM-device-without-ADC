# importing os module for environment variables
import os
# importing necessary functions from dotenv library
from dotenv import load_dotenv

__all__ = [
    "HP_TOKEN"
]

# loading variables from .env file
load_dotenv()

# getenv
HP_TOKEN = os.getenv("HP_TOKEN")
