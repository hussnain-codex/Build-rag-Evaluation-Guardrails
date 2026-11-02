
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

# Load a small, widely used embedding model
MODEL_NAME = EMBEDDING_MODEL_NAME

_model_cache = None


def get_embedding_model():
    """Load the SentenceTransformer model once and reuse it on later calls."""

    global _model_cache

    if _model_cache is None:
        _model_cache = SentenceTransformer(MODEL_NAME)

    return _model_cache


def create_embeddings(texts, show_progress_bar=False):
    """Convert a list of text strings into numerical vectors."""

    model = get_embedding_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar
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