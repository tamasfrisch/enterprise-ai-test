
import sys
import csv
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import anthropic
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from model_config import MAX_OUTPUT_TOKENS, TOP_K
from retrieve import load_documents, retrieve, tokenize

load_dotenv(PROJECT_DIR / ".env")

client = anthropic.Anthropic()

MODEL = "claude-sonnet-5-5"

CORPUS_DIR = PROJECT_DIR / "nordstar_corpus"
RESULTS_DIR = PROJECT_DIR / "results"
QUESTIONS_FILE = CORPUS_DIR / "benchmark_questions.csv"
OUTPUT_FILE = RESULTS_DIR / "anthropic_repetitions_3.csv"


def call_model(prompt):
    """Call Claude and return text, usage, and elapsed time."""
    start = time.perf_counter()

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        cache_control={"type": "ephemeral"},
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    elapsed = time.perf_counter() - start

    text = "\n".join(
        block.text
        for block in response.content
        if block.type == "text"
    )

    usage = response.usage

    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0

    # Anthropic reports cache usage separately.
    cache_read_tokens = getattr(
        usage, "cache_read_input_tokens", 0
    ) or 0

    cache_creation_tokens = getattr(
        usage, "cache_creation_input_tokens", 0
    ) or 0

    status = (
        "completed"
        if response.stop_reason == "end_turn"
        else "incomplete"
    )

    return {
        "text": text,
        "status": status,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "elapsed": elapsed,
    }


def run_experiment(question, chunks, bm25, architecture, repetition):
    question_id = question["question_id"]
    question_text = question["question"]
    timestamp = datetime.now(timezone.utc).isoformat()

    retrieval_seconds = 0.0
    api_seconds = 0.0
    retrieved = []
    calls = []
    answer = ""
    error = ""

    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_creation = 0
    statuses = []

    start_total = time.perf_counter()

    try:
        if architecture == "simple_chat":
            prompt = (
                "Answer the following question. If you do not have "
                "enough information, say so rather than guessing.\n\n"
                f"Question: {question_text}"
            )
            calls.append(call_model(prompt))

        else:
            retrieval_start = time.perf_counter()
            retrieved = retrieve(
                question_text, chunks, bm25, top_k=TOP_K
            )
            retrieval_seconds = time.perf_counter() - retrieval_start

            context = "\n\n".join(
                f"[Source: {item['document']}]\n{item['text']}"
                for item in retrieved
            )

            if architecture == "rag":
                prompt = f"""
Answer the question using only the supplied company documents.
If the documents do not contain enough information, say so.
Cite the source document names in your answer.

DOCUMENTS:
{context}

QUESTION:
{question_text}
"""
                calls.append(call_model(prompt))

            elif architecture == "bounded_two_stage":
                extraction_prompt = f"""
Extract the facts and conditions needed to answer the question.
Use only the supplied documents. Do not infer missing information.
Include relevant thresholds, exceptions, dates, and source document names.

DOCUMENTS:
{context}

QUESTION:
{question_text}
"""
                first = call_model(extraction_prompt)
                calls.append(first)

                if first["status"] == "completed" and first["text"]:
                    final_prompt = f"""
Answer the original question using only the extracted facts below.
Be precise and include relevant conditions and source document names.
Do not add facts that are not in the extracted material.

QUESTION:
{question_text}

EXTRACTED FACTS:
{first['text']}
"""
                    calls.append(call_model(final_prompt))
                else:
                    answer = (
                        "The first stage did not complete successfully; "
                        "no final answer was generated."
                    )

        # Aggregate usage across every model call.
        for call in calls:
            total_input += call["input_tokens"]
            total_output += call["output_tokens"]
            total_cache_read += call["cache_read_tokens"]
            total_cache_creation += call["cache_creation_tokens"]
            api_seconds += call["elapsed"]
            statuses.append(call["status"])

        if not answer and calls:
            answer = calls[-1]["text"]

        if not statuses:
            status = "failed"
        elif all(s == "completed" for s in statuses):
            status = "completed"
        else:
            status = "incomplete"

    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

        # Preserve usage from calls completed before an error.
        for call in calls:
            total_input += call["input_tokens"]
            total_output += call["output_tokens"]
            total_cache_read += call["cache_read_tokens"]
            total_cache_creation += call["cache_creation_tokens"]
            api_seconds += call["elapsed"]
            statuses.append(call["status"])

    end_to_end_seconds = time.perf_counter() - start_total

    retrieved_docs = sorted({
        item["document"] for item in retrieved
    })

    # Anthropic reports cached input separately from uncached input.
    billable_input_tokens = (
        total_input + total_cache_read + total_cache_creation
    )

    return {
        "timestamp": timestamp,
        "provider": "Anthropic",
        "model": MODEL,
        "architecture": architecture,
        "question_id": question_id,
        "repetition": repetition,
        "status": status,
        "answer": answer,
        "error": error,
        "input_tokens": total_input,
        "cache_read_input_tokens": total_cache_read,
        "cache_creation_input_tokens": total_cache_creation,
        "billable_input_tokens": billable_input_tokens,
        "output_tokens": total_output,
        "reasoning_tokens": 0,
        "total_tokens": billable_input_tokens + total_output,
        "model_calls": len(calls),
        "retrieval_latency_seconds": round(retrieval_seconds, 4),
        "api_latency_seconds": round(api_seconds, 4),
        "end_to_end_latency_seconds": round(end_to_end_seconds, 4),
        "retrieved_documents": ";".join(retrieved_docs),
        "estimated_cost_usd": "",
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. "
            "Rename or back it up before running again."
        )

    questions = pd.read_csv(QUESTIONS_FILE).fillna("")
    chunks = load_documents()

    bm25 = BM25Okapi([
        tokenize(item["text"]) for item in chunks
    ])

    architectures = [
        "simple_chat",
        "rag",
        "bounded_two_stage",
    ]

    repetitions = 3
    rows = []

    print(f"Provider: Anthropic")
    print(f"Model: {MODEL}")
    print(f"Questions: {len(questions)}")
    print(f"Architectures: {len(architectures)}")
    print(f"Repetitions: {repetitions}")
    print("Starting 27 observations...\n")

    for repetition in range(1, repetitions + 1):
        for _, question in questions.iterrows():
            for architecture in architectures:
                print(
                    f"Repetition {repetition}/{repetitions}: "
                    f"{question['question_id']} / {architecture}..."
                )

                result = run_experiment(
                    question, chunks, bm25,
                    architecture, repetition
                )
                rows.append(result)

                print(
                    f"  Status: {result['status']} | "
                    f"Tokens: {result['total_tokens']} | "
                    f"Calls: {result['model_calls']} | "
                    f"Time: {result['end_to_end_latency_seconds']}s"
                )

                if result["error"]:
                    print(f"  Error: {result['error']}")

                # Save after every observation.
                pd.DataFrame(rows).to_csv(
                    OUTPUT_FILE,
                    index=False,
                    quoting=csv.QUOTE_MINIMAL
                )

    print("\nAnthropic experiment finished.")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Completed observations: {len(rows)}")


if __name__ == "__main__":
    main()
