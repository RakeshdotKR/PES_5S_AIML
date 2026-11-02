import os 
import time
import json

from dotenv import load_dotenv #.env
from google import genai
from google.genai import types 

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found")

client = genai.Client(api_key=GEMINI_API_KEY)

ping = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly three words:API key works."
)
print(f"[PING]{ping.text.strip()}")