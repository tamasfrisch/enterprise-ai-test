
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-6-luna"

start_time = time.time()

response = client.responses.create(
    model=MODEL,
    reasoning={"effort": "none"},
    input="Say hello in exactly five words.",
    max_output_tokens=100
)

latency = time.time() - start_time

print("Model:", MODEL)
print("Status:", response.status)
print("Response:", repr(response.output_text))

print("\nUsage:")
print("Input tokens:", response.usage.input_tokens)
print("Output tokens:", response.usage.output_tokens)
print("Total tokens:", response.usage.total_tokens)

print(f"Latency: {latency:.3f} seconds")
