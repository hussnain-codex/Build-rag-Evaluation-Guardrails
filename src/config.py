"""Central configuration: shared paths, model names and defaults.

Every other module imports PROJECT_ROOT and model names from here instead of
recomputing Path(__file__).resolve().parent.parent on its own.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PDF_FOLDER = PROJECT_ROOT / "pdfs"
EVAL_FOLDER = PROJECT_ROOT / "eval"
RESULTS_FOLDER = PROJECT_ROOT / "results"
STORE_FOLDER = PROJECT_ROOT / "vector_store"

FAISS_INDEX_PATH = STORE_FOLDER / "faiss.index"
CHUNKS_METADATA_PATH = STORE_FOLDER / "chunks.json"
EVAL_DATASET_CSV = EVAL_FOLDER / "eval.csv"
ADVERSARIAL_PROMPTS_CSV = EVAL_FOLDER / "adversarial_prompts.csv"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL_NAME = "gemini-3.1-flash-lite"
# NOTE: "gemini-3.8-flash" is the model this project was originally wired to.
# It exists on this account (confirmed via models.get) but generateContent
# calls to it hang indefinitely in this environment (verified with both
# langchain-google-genai and the native google-genai SDK, and via a raw
# REST call with an explicit timeout -- all hung past 45-120s). Other
# models (gemini-3.1-flash-lite, gemini-flash-latest) respond normally
# (1-3s, or a clean 503 under load), so this looks like a model-specific
# issue on Google's side rather than a bug in this codebase. Swap this
# back to "gemini-3.8-flash" once that model responds normally again.

DEFAULT_TOP_K = 4
RERANK_CANDIDATE_POOL = 20

ENV_FILE = PROJECT_ROOT / ".env"
