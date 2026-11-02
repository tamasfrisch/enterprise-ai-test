
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model_config import MODEL, REASONING_EFFORT, MAX_OUTPUT_TOKENS
from retrieve import load_documents, retrieve, tokenize

load_dotenv(PROJECT_DIR / ".env")
client = OpenAI()

question = (
    "What documentation is required after an emergency purchase, "
    "and when must it be provided?"
)

chunks = load_documents()
bm25 = BM25Okapi([
    tokenize(item["text"]) for item in chunks
])

retrieved = retrieve(question, chunks, bm25, top_k=5)

# Move the paragraph containing the specific deadline to the top.
relevant = [
    item for item in retrieved
    if "five business days" in item["text"].lower()
]
others = [
    item for item in retrieved
    if "five business days" not in item["text"].lower()
]
reordered = relevant + others

print("Passage order:")
for i, item in enumerate(reordered, 1):
    print(f"{i}. {item['document']} — {item['text'][:100]}...")

context = "\n\n".join(
    f"[Source: {item['document']}]\n{item['text']}"
    for item in reordered
)

prompt = f"""
Answer the question using only the supplied company documents.

Identify all required documentation elements and the deadline.
Distinguish emergency-purchase documentation requirements from
general record-retention requirements.
Do not omit facts that are explicitly stated in the documents.

DOCUMENTS:
{context}

QUESTION:
{question}
"""

response = client.responses.create(
    model=MODEL,
    input=prompt,
    reasoning={"effort": REASONING_EFFORT},
    max_output_tokens=MAX_OUTPUT_TOKENS,
)

print("\nStatus:", response.status)
print("\nAnswer:\n")
print(response.output_text)

if response.usage:
    print("\nInput tokens:", response.usage.input_tokens)
    print("Output tokens:", response.usage.output_tokens)
    print("Total tokens:", response.usage.total_tokens)
