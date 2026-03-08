from google.genai import Client
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = Client(api_key=GEMINI_API_KEY)


class GeminiLLM:
    def __init__(self, model_name="gemini-flash-latest"):
        self.model = model_name

    def __call__(self, prompt, temperature=0.0):
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": 1024,
            },
        )
        return response.text