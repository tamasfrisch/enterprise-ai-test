
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

from sys import path
path.insert(0, str(PROJECT_DIR / "src"))

from model_config import MODEL, REASONING_EFFORT, MAX_OUTPUT_TOKENS

client = OpenAI()

evidence = """
The responsible employee must document the reason for the emergency,
the purchase amount, the supplier, and the circumstances requiring
urgent action. Emergency purchases must be documented within five
business days of the purchase. Outstanding approvals must then be
obtained in accordance with the applicable thresholds.
"""

question = (
    "What documentation is required after an emergency purchase, "
    "and when must it be provided?"
)

prompt = f"""
Answer the question using the evidence below.

Evidence:
{evidence}

Question:
{question}

State all four required elements:
- reason for the emergency
- purchase amount
- supplier and circumstances requiring urgent action
- deadline

Do not claim that information is missing if it appears in the evidence.
"""

response = client.responses.create(
    model=MODEL,
    input=prompt,
    reasoning={"effort": REASONING_EFFORT},
    max_output_tokens=MAX_OUTPUT_TOKENS,
)

print("Status:", response.status)
print("\nAnswer:\n")
print(response.output_text)

if response.usage:
    print("\nInput tokens:", response.usage.input_tokens)
    print("Output tokens:", response.usage.output_tokens)
    print("Total tokens:", response.usage.total_tokens)
