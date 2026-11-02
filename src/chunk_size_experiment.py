"""README deliverable: chunk-size comparison table.

Rebuilds the vector store with a few different (chunk_size, chunk_overlap)
settings, in isolated in-memory indexes (the real vector_store/ on disk is
left untouched), and re-runs the retrieval Hit Rate/MRR/Precision eval for
each so the README table is backed by real numbers.
"""

import json

import faiss
import numpy as np
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EVAL_DATASET_CSV, DEFAULT_TOP_K, RESULTS_FOLDER
from load_pdfs import load_pdfs
from embeddings import create_embeddings

CHUNK_CONFIGS = [
    {"chunk_size": 300, "chunk_overlap": 30},
    {"chunk_size": 500, "chunk_overlap": 50},  # current default
    {"chunk_size": 800, "chunk_overlap": 80},
    {"chunk_size": 1000, "chunk_overlap": 100},
]


def build_chunks(documents, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = []
    for document in documents:
        for chunk_index, chunk_text in enumerate(splitter.split_text(document["text"])):
            chunks.append({
                "text": chunk_text,
                "metadata": {**document["metadata"], "chunk_index": chunk_index},
            })
    return chunks


def build_in_memory_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    vectors = np.asarray(create_embeddings(texts), dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve_from_index(index, chunks, question, top_k=DEFAULT_TOP_K):
    query_vector = np.asarray(create_embeddings([question]), dtype="float32")
    k = min(top_k, index.ntotal)
    if k == 0:
        return []
    scores, indices = index.search(query_vector, k)
    results = []
    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index == -1:
            continue
        chunk = chunks[int(chunk_index)]
        results.append({
            "source": chunk["metadata"]["source"],
            "page": chunk["metadata"]["page"],
            "score": float(score),
        })
    return results


def evaluate_config(documents, dataset, chunk_size, chunk_overlap, top_k=DEFAULT_TOP_K):
    chunks = build_chunks(documents, chunk_size, chunk_overlap)
    index = build_in_memory_index(chunks)

    answerable = dataset[dataset["answerable"].astype(str).str.lower() == "true"]
    total = len(answerable)
    hits = 0
    reciprocal_ranks = []

    for _, item in answerable.iterrows():
        expected_source = item["file"]
        expected_page = int(item["page"])
        results = retrieve_from_index(index, chunks, item["question"], top_k=top_k)

        rank = None
        for position, result in enumerate(results, start=1):
            if result["source"] == expected_source and result["page"] == expected_page:
                rank = position
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_chunks": len(chunks),
        "total_questions": total,
        "hit_rate": hits / total if total else 0,
        "mrr": sum(reciprocal_ranks) / total if total else 0,
    }


def run_experiment():
    documents = load_pdfs()
    dataset = pd.read_csv(EVAL_DATASET_CSV, keep_default_na=False)
    dataset["page"] = pd.to_numeric(dataset["page"], errors="coerce")

    rows = []
    for config in CHUNK_CONFIGS:
        print(f"Testing chunk_size={config['chunk_size']}, "
              f"chunk_overlap={config['chunk_overlap']} ...")
        result = evaluate_config(
            documents, dataset, config["chunk_size"], config["chunk_overlap"]
        )
        rows.append(result)
        print(
            f"  -> {result['total_chunks']} chunks | "
            f"Hit Rate@4 = {result['hit_rate']:.2%} | MRR = {result['mrr']:.4f}"
        )

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_FOLDER / "chunk_size_comparison.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)
    print("\nSaved chunk-size comparison to:", out_path)
    return rows


if __name__ == "__main__":
    run_experiment()
