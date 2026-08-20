import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is not configured.")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
)

models = client.models.list()

print("Available models:")
for model in models.data:
    print(model.id)

