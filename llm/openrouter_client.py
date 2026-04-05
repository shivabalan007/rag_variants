import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def llm(prompt, temperature=0.1):

    completion = client.chat.completions.create(
        model="deepseek/deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=800,
    )
    
    return completion.choices[0].message.content


    # mistralai/mistral-small-3.1-24b-instruct
    # meta-llama/llama-3-8b-instruct
    # meta-llama/llama-3-70b-instruct
    # mistralai/mistral-large