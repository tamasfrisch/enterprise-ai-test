
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise ValueError(
        "ANTHROPIC_API_KEY was not found in the .env file."
    )

client = anthropic.Anthropic(api_key=api_key)

MODEL = "claude-sonnet-5-5"

start = time.perf_counter()

response = client.messages.create(
    model=MODEL,
    max_tokens=300,
    messages=[
        {
            "role": "user",
            "content": "Reply with: Anthropic API connection successful."
        }
    ],
)

elapsed = time.perf_counter() - start

print("Provider: Anthropic")
print("Model:", response.model)
print("Status:", response.stop_reason)
print("Response:")

for block in response.content:
    if block.type == "text":
        print(block.text)

print("\nInput tokens:", response.usage.input_tokens)
print("Output tokens:", response.usage.output_tokens)
print("Total tokens:",
      response.usage.input_tokens + response.usage.output_tokens)
print("Latency:", round(elapsed, 3), "seconds")
