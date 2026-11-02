
from pathlib import Path
import pandas as pd

from retrieve import load_documents, tokenize, BM25Okapi, retrieve

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

questions = pd.read_csv(DATA_DIR / "benchmark_questions.csv")

chunks = load_documents()
bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

results = []

for _, question in questions.iterrows():
    expected_sources = [
        name.strip()
        for name in question["source_document"].split(";")
    ]

    retrieved = retrieve(
        question["question"], chunks, bm25, top_k=5
    )

    ranked_sources = [r["document"] for r in retrieved]

    row = {
        "question_id": question["question_id"],
    }

    for k in [1, 3, 5]:
        found = any(
            source in ranked_sources[:k]
            for source in expected_sources
        )
        row[f"hit_at_{k}"] = int(found)

    reciprocal_rank = 0.0
    for rank, source in enumerate(ranked_sources, start=1):
        if source in expected_sources:
            reciprocal_rank = 1 / rank
            break

    row["reciprocal_rank"] = reciprocal_rank
    row["retrieved_documents"] = "; ".join(ranked_sources)

    results.append(row)

df = pd.DataFrame(results)

print("\nIndividual results:")
print(df.to_string(index=False))

print("\nAggregate metrics:")
for metric in ["hit_at_1", "hit_at_3", "hit_at_5", "reciprocal_rank"]:
    print(f"{metric}: {df[metric].mean():.3f}")

output_file = PROJECT_DIR / "results" / "retrieval_baseline.csv"
df.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
