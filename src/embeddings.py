
from sentence_transformers import SentenceTransformer


# Load a small, widely used embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings(texts):
    """Convert a list of text strings into numerical vectors."""

    model = SentenceTransformer(MODEL_NAME)

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return vectors


if __name__ == "__main__":
    sample_texts = [
        "A course lasts twelve weeks.",
        "The course duration is twelve weeks.",
        "Python is a programming language."
    ]

    embeddings = create_embeddings(sample_texts)

    print("Number of texts:", len(sample_texts))
    print("Embedding matrix shape:", embeddings.shape)
    print("First vector dimensions:", len(embeddings[0]))