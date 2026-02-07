from openai import OpenAI
import os
from dotenv import load_dotenv
from google import genai
load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'), 
    base_url=os.getenv('OPENAI_API_BASE')
)
gemini_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))


