
import json

import faiss
import numpy as np

from chunking import create_chunks
from load_pdfs import load_pdfs
from embeddings import create_embeddings
from config import STORE_FOLDER, FAISS_INDEX_PATH, CHUNKS_METADATA_PATH

INDEX_PATH = FAISS_INDEX_PATH
METADATA_PATH = CHUNKS_METADATA_PATH


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

    embeddings = create_embeddings(texts, show_progress_bar=True)
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