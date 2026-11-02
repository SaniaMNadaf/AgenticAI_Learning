import os
import time
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

client = genai.Client(api_key=GEMINI_API_KEY)

ping = client.models.generate_content(
    model="gemini-3.6-flash",
    contents="Reply with exactly three words:API key works."
)
print(f"[PING]{ping.text.strip()}")