import os
import openai
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_key
)

def generate_responses(prompt, temperature, num_responses=5):
    responses = []

    for i in range(num_responses):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        responses.append(response.choices[0].message.content)

    return responses


def save_results(prompt, temperature, responses):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{temperature}_{timestamp}.json"

    data = {
        "prompt": prompt,
        "temperature": temperature,
        "responses": responses
    }

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


temperatures = [0.0, 0.3, 0.7, 1.0]

prompts = [
    "Describe a world where apples are used as currency.",
    "Write a short story about a golden apple.",
    "Create a business plan for a luxury apple orchard."
]

for temp in temperatures:
    for prompt in prompts:
        responses = generate_responses(prompt, temp)
        save_results(prompt, temp, responses)
