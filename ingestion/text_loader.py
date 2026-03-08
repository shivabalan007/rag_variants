from ingestion.base import Document

def load_txt(path: str):
    documents = []

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if text.strip():
        documents.append(
            Document(
                text = text.strip(),
                metadata={
                    "source": path,
                    "doc_type":"txt"
                }
            )
        )
    return documents