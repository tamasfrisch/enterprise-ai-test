
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model_config import MODEL, REASONING_EFFORT, MAX_OUTPUT_TOKENS
from retrieve import load_documents, retrieve
from rank_bm25 import BM25Okapi

load_dotenv(PROJECT_DIR / ".env")
client = OpenAI()

QUESTION = (
    "What documentation is required after an emergency purchase, "
    "and when must it be provided?"
)

chunks = load_documents()
bm25 = BM25Okapi([
    item["text"].lower().split() for item in chunks
])

retrieved = retrieve(QUESTION, chunks, bm25, top_k=5)

context = "\n\n".join(
    f"[Source: {item['document']}]\n{item['text']}"
    for item in retrieved
)

prompt = f"""
Answer the question using only the supplied company documents.

IMPORTANT:
1. Identify every distinct requirement that directly answers the question.
2. Check the evidence for the required documentation contents.
3. Check the evidence for the deadline.
4. Do not confuse the deadline for documenting an emergency purchase
   with general document-retention requirements.
5. Before writing the answer, verify that you have addressed both
   parts of the question.
6. If the documents contain the information, do not say it is missing.

Give a concise answer and cite the source document.

DOCUMENTS:
{context}

QUESTION:
{QUESTION}
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
