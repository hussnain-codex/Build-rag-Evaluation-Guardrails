
from pathlib import Path
from pypdf import PdfReader

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder containing PDF documents
PDF_FOLDER = PROJECT_ROOT / "pdfs"


def load_pdfs():
    """Extract text and preserve each PDF's filename and page number."""

    documents = []

    # Find every PDF in the pdfs folder
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the pdfs folder.")
        return documents

    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")

        reader = PdfReader(str(pdf_path))

        # pypdf page indexes start at 0
        # Human-readable page numbers start at 1
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if not text.strip():
                continue

            documents.append({
                "text": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_index + 1
                }
            })

    return documents


if __name__ == "__main__":
    pages = load_pdfs()

    print(f"\nTotal pages extracted: {len(pages)}")

    for item in pages[:3]:
        print("\n------------------------------")
        print("Source:", item["metadata"]["source"])
        print("Page:", item["metadata"]["page"])
        print("Text preview:", item["text"][:300])