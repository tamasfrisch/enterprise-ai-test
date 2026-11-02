
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

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_DIR / "nordstar_corpus" / "corpus"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "experiment_results.csv"

RESULTS_DIR.mkdir(exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

QUESTION_ID = "TASK-01"
QUESTION = (
    "Who must approve a purchase worth EUR 75000 "
    "under Nordstar's procurement policy?"
)

def main():

    workflow_start = time.time()

    # 1. Retrieve relevant passages
    chunks = load_documents()
    bm25 = BM25Okapi([
        tokenize(chunk["text"]) for chunk in chunks
    ])

    retrieval_start = time.time()

    retrieval_results = retrieve(
        QUESTION, chunks, bm25, top_k=TOP_K
    )

    retrieval_latency = time.time() - retrieval_start

    context = "\n\n".join(
        f"Source: {r['document']}\n{r['text']}"
        for r in retrieval_results
    )

    # 2. First model call: fact extraction
    extraction_prompt = f"""
You are a policy analysis assistant.

Extract the exact approval requirements relevant to this question.
Use only the supplied documents.
List the approval authorities and relevant monetary thresholds.
Do not write the final answer yet.

DOCUMENTS:
{context}

QUESTION:
{QUESTION}
"""

    call1_start = time.time()

    extraction = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=extraction_prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    call1_latency = time.time() - call1_start

    if extraction.status != "completed":
        raise RuntimeError(
            f"Fact extraction did not complete: {extraction.status}"
        )

    extracted_facts = extraction.output_text

    # 3. Second model call: final answer
    final_prompt = f"""
Answer the original question using the extracted facts.
Do not introduce new facts.
Be concise and identify the source document.

QUESTION:
{QUESTION}

EXTRACTED FACTS:
{extracted_facts}
"""

    call2_start = time.time()

    final_response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=final_prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    call2_latency = time.time() - call2_start

    if final_response.status != "completed":
        raise RuntimeError(
            f"Final answer did not complete: {final_response.status}"
        )

    # 4. Aggregate usage across both model calls
    total_input = (
        extraction.usage.input_tokens
        + final_response.usage.input_tokens
    )

    total_output = (
        extraction.usage.output_tokens
        + final_response.usage.output_tokens
    )

    reasoning_tokens = (
        (extraction.usage.output_tokens_details.reasoning_tokens or 0)
        + (final_response.usage.output_tokens_details.reasoning_tokens or 0)
    )

    total_tokens = total_input + total_output

    workflow_latency = time.time() - workflow_start

    # 5. Save results
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "OpenAI",
        "model": MODEL,
        "architecture": "agentic",
        "question_id": QUESTION_ID,
        "repetition": 1,
        "status": final_response.status,
        "answer": final_response.output_text,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": round(workflow_latency, 3),
        "model_calls": 2,
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

    # 6. Display results
    print("Agentic experiment completed.")
    print("Model:", MODEL)
    print("Question:", QUESTION_ID)

    print("\nRetrieved documents:")
    for r in retrieval_results:
        print("-", r["document"])

    print("\nExtracted facts:")
    print(extracted_facts)

    print("\nFinal answer:")
    print(final_response.output_text)

    print("\nUsage across both calls:")
    print("Input tokens:", total_input)
    print("Output tokens:", total_output)
    print("Reasoning tokens:", reasoning_tokens)
    print("Total tokens:", total_tokens)
    print("Model calls: 2")

    print("\nLatency:")
    print("Retrieval:", round(retrieval_latency, 3))
    print("Call 1:", round(call1_latency, 3))
    print("Call 2:", round(call2_latency, 3))
    print("End-to-end:", round(workflow_latency, 3))

    print("\nSaved to:", RESULTS_FILE)


if __name__ == "__main__":
    main()
