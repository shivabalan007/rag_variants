from pypdf import PdfReader
from ingestion.base import Document


def load_pdf(path: str):
    reader = PdfReader(path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()

        if not text or not text.split():
            continue

        doc = Document(
            text=text.strip(),
            metadata={
                "source": path,
                "page": page_num + 1,
                "doc_type": "pdf"
            }
        )
        documents.append(doc)

    return documents