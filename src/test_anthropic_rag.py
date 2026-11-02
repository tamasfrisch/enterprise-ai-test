
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from retrieve import load_documents, retrieve, tokenize

load_dotenv(PROJECT_DIR / ".env")
client = anthropic.Anthropic()

MODEL = "claude-sonnet-5-5"

question = (
    "What documentation is required after an emergency purchase, "
    "and when must it be provided?"
)

chunks = load_documents()
bm25 = BM25Okapi([
    tokenize(item["text"]) for item in chunks
])

retrieved = retrieve(question, chunks, bm25, top_k=5)

context = "\n\n".join(
    f"[Source: {item['document']}]\n{item['text']}"
    for item in retrieved
)

prompt = f"""
Answer the question using only the supplied company documents.
If the documents do not contain enough information, say so.
Cite the source document names in your answer.

DOCUMENTS:
{context}

QUESTION:
{question}
"""

start = time.perf_counter()

response = client.messages.create(
    model=MODEL,
    max_tokens=1000,
    messages=[{"role": "user", "content": prompt}],
)

elapsed = time.perf_counter() - start

answer = "\n".join(
    block.text
    for block in response.content
    if block.type == "text"
)

print("Provider: Anthropic")
print("Model:", response.model)
print("Status:", response.stop_reason)
print("\nAnswer:\n")
print(answer)

print("\nInput tokens:", response.usage.input_tokens)
print("Output tokens:", response.usage.output_tokens)
print("Total tokens:",
      response.usage.input_tokens + response.usage.output_tokens)
print("Latency:", round(elapsed, 3), "seconds")
