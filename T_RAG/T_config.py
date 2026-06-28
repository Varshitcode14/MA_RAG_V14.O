import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEYS").split(",")[0]

MODEL_NAME = "llama-3.3-70b-versatile"

TOP_K = 5

TEMPERATURE = 0