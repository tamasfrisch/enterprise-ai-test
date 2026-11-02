
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
    MAX_OUTPUT_TOKENS
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "experiment_results.csv"

RESULTS_DIR.mkdir(exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=api_key)

# --------------------------------------------------
# Benchmark question
# --------------------------------------------------

QUESTION_ID = "TASK-01"

QUESTION = (
    "Who must approve a purchase worth EUR 75000 "
    "under Nordstar's procurement policy?"
)

# --------------------------------------------------
# Execute one experiment
# --------------------------------------------------

def run_experiment():

    architecture = "simple_chat"
    repetition = 1

    print("Starting experiment...")
    print("Model:", MODEL)
    print("Architecture:", architecture)
    print("Question:", QUESTION_ID)

    start_time = time.time()

    response = client.responses.create(
        model=MODEL,
        reasoning={"effort": REASONING_EFFORT},
        input=QUESTION,
        max_output_tokens=MAX_OUTPUT_TOKENS
    )

    latency = time.time() - start_time

    usage = response.usage

    reasoning_tokens = 0

    if usage.output_tokens_details:
        reasoning_tokens = (
            usage.output_tokens_details.reasoning_tokens or 0
        )

    # Prepare result record
    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "OpenAI",
        "model": MODEL,
        "architecture": architecture,
        "question_id": QUESTION_ID,
        "repetition": repetition,
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

    # Save result to CSV
    file_exists = RESULTS_FILE.exists()

    with RESULTS_FILE.open(
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=result.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    # Display result
    print("\nExperiment completed.")
    print("Status:", response.status)
    print("Answer:", response.output_text)
    print("Input tokens:", usage.input_tokens)
    print("Output tokens:", usage.output_tokens)
    print("Reasoning tokens:", reasoning_tokens)
    print("Latency:", round(latency, 3), "seconds")
    print("Saved to:", RESULTS_FILE)


if __name__ == "__main__":
    run_experiment()
