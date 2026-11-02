
from pathlib import Path
import re
from rank_bm25 import BM25Okapi

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_DIR / "data" / "corpus"
QUESTIONS_FILE = PROJECT_DIR / "data" / "benchmark_questions.csv"

def tokenize(text):
    """Simple lowercase word tokenizer."""
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

def load_documents():
    """Load text files and split them into paragraph-based chunks."""
    chunks = []

    for file_path in sorted(CORPUS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")

        # Split documents into paragraphs
        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        for i, paragraph in enumerate(paragraphs):
            chunks.append({
                "document": file_path.name,
                "chunk_id": i,
                "text": paragraph
            })

    return chunks

def retrieve(query, chunks, bm25, top_k=5):
    """Return the top-k most relevant chunks."""
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:
        results.append({
            **chunks[index],
            "score": float(scores[index])
        })

    return results

def main():
    import pandas as pd

    chunks = load_documents()

    if not chunks:
        raise RuntimeError(f"No documents found in {CORPUS_DIR}")

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    questions = pd.read_csv(QUESTIONS_FILE)

    print(f"Loaded {len(chunks)} chunks from the corpus.")
    print(f"Testing {len(questions)} benchmark questions.\n")

    for _, question in questions.iterrows():
        print("=" * 75)
        print(f"{question['question_id']}: {question['question']}")
        print("-" * 75)

        results = retrieve(
            question["question"],
            chunks,
            bm25,
            top_k=5
        )

        for rank, result in enumerate(results, start=1):
            print(
                f"\n{rank}. {result['document']} "
                f"(chunk {result['chunk_id']}, "
                f"score={result['score']:.3f})"
            )
            print(result["text"][:500])

        print()

if __name__ == "__main__":
    main()
