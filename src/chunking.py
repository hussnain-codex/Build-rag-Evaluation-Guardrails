
from load_pdfs import load_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_chunks(documents):
    """Split PDF pages into smaller chunks while preserving metadata."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True
    )
    chunks = []

    for document in documents:
        split_texts = splitter.split_text(document["text"])

        for chunk_index, chunk_text in enumerate(split_texts):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **document["metadata"],
                    "chunk_index": chunk_index,
                }
            })

    return chunks


if __name__ == "__main__":
    documents = load_pdfs()
    chunks = create_chunks(documents)

    print(f"Total PDF pages: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    for chunk in chunks[:3]:
        print("\n------------------------------")
        print("Source:", chunk["metadata"]["source"])
        print("Page:", chunk["metadata"]["page"])
        print("Chunk:", chunk["metadata"]["chunk_index"])
        print("Text:", chunk["text"][:200])