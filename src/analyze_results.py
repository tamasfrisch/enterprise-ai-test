
from pathlib import Path
import pandas as pd

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

FILES = {
    "openai": RESULTS / "openai_repetitions_3.csv",
    "anthropic": RESULTS / "anthropic_repetitions_3.csv",
}

PRICING = {
    "gpt-6-luna": {
        "input": 0.10,
        "output": 0.50,
    },
    "claude-sonnet-5-5": {
        "input": 2.00,
        "output": 10.00,
    },
}

# Number of ground-truth criteria in each benchmark task.
CRITERIA_PER_TASK = {
    "TASK-01": 3,
    "TASK-02": 4,
    "TASK-03": 3,
}

# Based on the manual answer review completed for both providers.
# Each RAG and two-stage answer met every criterion.
# Simple chat appropriately abstained, but did not answer the task.
SUCCESS_BY_ARCHITECTURE = {
    "simple_chat": 0,
    "rag": 1,
    "bounded_two_stage": 1,
}

# ------------------------------------------------------------
# Load and normalize experiment data
# ------------------------------------------------------------

frames = []

for provider, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    df = pd.read_csv(path)
    df["provider"] = provider

    # Normalize input-token field for cost calculation.
    if provider == "anthropic":
        df["cost_input_tokens"] = df["billable_input_tokens"]
    else:
        df["cost_input_tokens"] = df["input_tokens"]

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# ------------------------------------------------------------
# Add manually reviewed task-quality scores
# ------------------------------------------------------------

data["criteria_total"] = data["question_id"].map(CRITERIA_PER_TASK)

data["task_success"] = data["architecture"].map(
    SUCCESS_BY_ARCHITECTURE
)

data["criteria_satisfied"] = (
    data["criteria_total"] * data["task_success"]
)

# ------------------------------------------------------------
# Calculate API costs, if verified rates are configured
# ------------------------------------------------------------

def calculate_cost(row):
    model = row["model"]
    rates = PRICING.get(model)

    if not rates:
        return None

    input_rate = rates["input"]
    output_rate = rates["output"]

    if input_rate is None or output_rate is None:
        return None

    input_cost = (
        row["cost_input_tokens"] * input_rate / 1_000_000
    )
    output_cost = (
        row["output_tokens"] * output_rate / 1_000_000
    )

    return input_cost + output_cost


data["calculated_cost_usd"] = data.apply(
    calculate_cost, axis=1
)

# Existing script estimate retained for comparison.
if "estimated_cost_usd" in data.columns:
    data["existing_estimated_cost_usd"] = data[
        "estimated_cost_usd"
    ]

# ------------------------------------------------------------
# Save consolidated run-level dataset
# ------------------------------------------------------------

data.to_csv(
    RESULTS / "consolidated_experiment_results.csv",
    index=False
)

# ------------------------------------------------------------
# Architecture-level descriptive summary
# ------------------------------------------------------------

summary = (
    data.groupby(["provider", "architecture"])
    .agg(
        runs=("question_id", "count"),
        mean_input_tokens=("input_tokens", "mean"),
        mean_output_tokens=("output_tokens", "mean"),
        mean_total_tokens=("total_tokens", "mean"),
        mean_latency_seconds=(
            "end_to_end_latency_seconds", "mean"
        ),
        sd_latency_seconds=(
            "end_to_end_latency_seconds", "std"
        ),
        successful_tasks=("task_success", "sum"),
        task_success_rate=("task_success", "mean"),
        criteria_satisfied=("criteria_satisfied", "sum"),
        criteria_total=("criteria_total", "sum"),
        calculated_cost_usd=("calculated_cost_usd", lambda x: x.sum(min_count=1)),
    )
    .reset_index()
)

summary["cost_per_successful_task_usd"] = (
    summary["calculated_cost_usd"]
    / summary["successful_tasks"].replace(0, pd.NA)
)

summary["cost_per_satisfied_criterion_usd"] = (
    summary["calculated_cost_usd"]
    / summary["criteria_satisfied"].replace(0, pd.NA)
)

summary.to_csv(
    RESULTS / "architecture_summary.csv",
    index=False
)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

print("\nCONSOLIDATED DATA")
print("Rows:", len(data))
print(data.groupby(["provider", "architecture"]).size())

print("\nARCHITECTURE SUMMARY")
print(summary.round(6).to_string(index=False))

print("\nFILES CREATED")
print("results/consolidated_experiment_results.csv")
print("results/architecture_summary.csv")

if data["calculated_cost_usd"].isna().any():
    print(
        "\nNOTE: Monetary costs were not calculated because "
        "verified model prices have not yet been configured."
    )
