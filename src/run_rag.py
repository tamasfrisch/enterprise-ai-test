
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import csv
import time
from datetime import datetime, timezone

from model_config import (
    MODEL,
    REASONING_EFFORT,
    MAX_OUTPUT_TOKENS,
    TOP_K
)

from retrieve import load_documents, tokenize, retrieve, BM25Okapi

# Paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_DIR / "nordstar_corpus" / "corpus"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "experiment_results.csv"

RESULTS_DIR.mkdir(exist_ok=True)

# API client
load_dotenv(PROJECT_DIR / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Benchmark question
QUESTION_ID = "TASK-01"
QUESTION = (
    "Who must approve a purchase worth EUR 75000 "
    "under Nordstar's procurement policy?"
)

def main():

    # Load and index documents
    chunks = load_documents()
    bm25 = BM25Okapi([
        tokenize(chunk["text"]) for chunk in chunks
    ])

    # Retrieve passages
    retrieval_results = retrieve(
        QUESTION, chunks, bm25, top_k=TOP_K
    )

    context = "\n\n".join(
        f"Source: {r['document']}\n{r['text']}"
        for r in retrieval_results
    )

    prompt = f"""
You are an assistant answering questions about Nordstar Industrial Systems GmbH.

Answer the question using only the supplied company documents.
If the documents do not contain the answer, say so.
Do not invent facts.
Identify the relevant source document in your answer.

COMPANY DOCUMENTS:
{context}

QUESTION:
{QUESTION}
"""

    print("Starting RAG experiment...")
    print("Model:", MODEL)
    print("Question:", QUESTION_ID)
    print("Retrieved passages:", len(retrieval_results))

    start = time.time()

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    latency = time.time() - start
    usage = response.usage

    reasoning_tokens = 0
    if usage.output_tokens_details:
        reasoning_tokens = (
            usage.output_tokens_details.reasoning_tokens or 0
        )

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "OpenAI",
        "model": MODEL,
        "architecture": "rag",
        "question_id": QUESTION_ID,
        "repetition": 1,
        "status": response.status,
        "answer": response.output_text,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": usage.total_tokens,
        "latency_seconds": round(latency, 3),
        "model_calls": 1,
        "estimated_cost_usd": ""
    }

    file_exists = RESULTS_FILE.exists()

    with RESULTS_FILE.open(
        "a", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=result.keys()
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print("\nRetrieved documents:")
    for r in retrieval_results:
        print("-", r["document"])

    print("\nAnswer:")
    print(response.output_text)

    print("\nUsage:")
    print("Input tokens:", usage.input_tokens)
    print("Output tokens:", usage.output_tokens)
    print("Reasoning tokens:", reasoning_tokens)
    print("Latency:", round(latency, 3), "seconds")

    print("\nSaved to:", RESULTS_FILE)


if __name__ == "__main__":
    main()
