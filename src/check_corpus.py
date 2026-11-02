from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"

documents = list(CORPUS_DIR.glob("*.txt"))

print(f"Corpus directory: {CORPUS_DIR}")
print(f"Number of documents: {len(documents)}")

for doc in sorted(documents):
    text = doc.read_text(encoding="utf-8")
    print(f"{doc.name}: {len(text)} characters")

questions = pd.read_csv(
    PROJECT_DIR / "data" / "benchmark_questions.csv"
)

ground_truth = pd.read_csv(
    PROJECT_DIR / "data" / "ground_truth.csv"
)

print(f"\nBenchmark questions: {len(questions)}")
print(f"Ground-truth criteria: {len(ground_truth)}")

assert len(documents) == 21, "Unexpected number of corpus documents"
assert len(questions) == 3, "Unexpected number of benchmark questions"

print("\nSUCCESS: Corpus and benchmark files are readable.")