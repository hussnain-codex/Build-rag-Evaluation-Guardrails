
import json
from pathlib import Path

import faiss
import numpy as np

from chunking import create_chunks
from load_pdfs import load_pdfs
from embeddings import create_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_FOLDER = PROJECT_ROOT / "vector_store"
INDEX_PATH = STORE_FOLDER / "faiss.index"
METADATA_PATH = STORE_FOLDER / "chunks.json"


def build_vector_store():
    """Create and save the FAISS index and its chunk metadata."""

    documents = load_pdfs()

    if not documents:
        print("No PDF text found. Add a readable PDF to pdfs/.")
        return

    chunks = create_chunks(documents)

    if not chunks:
        print("No chunks were created.")
        return

    texts = [chunk["text"] for chunk in chunks]

    embeddings = create_embeddings(texts)
    vectors = np.asarray(embeddings, dtype="float32")

    # Each vector has 384 dimensions for this embedding model
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    STORE_FOLDER.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print("\nVector store created successfully.")
    print("Total chunks indexed:", index.ntotal)
    print("Index location:", INDEX_PATH)
    print("Metadata location:", METADATA_PATH)


if __name__ == "__main__":
    build_vector_store()