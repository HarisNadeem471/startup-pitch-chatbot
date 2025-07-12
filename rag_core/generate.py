import os
import requests
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("TOGETHER_API_KEY")

def generate_answer(context_chunks, user_question):
    prompt = f"""You're an AI assistant analyzing a startup pitch deck.
Context:
{''.join(context_chunks)}

Question:
{user_question}

Answer:"""

    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.7
    }

    response = requests.post("https://api.together.xyz/inference", json=data, headers=headers)
    return response.json().get("output", {}).get("choices", [{}])[0].get("text", "⚠️ No response.")
