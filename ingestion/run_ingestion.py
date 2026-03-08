from ingestion.pdf_loader import load_pdf
from ingestion.text_loader import load_txt
from ingestion.csv_loader import load_csv

def ingest(path: str):
    if path.endswith(".pdf"):
        return load_pdf(path)
    elif path.endswith(".txt"):
        return load_txt(path)
    elif path.endswith(".csv"):
        return load_csv(path)
    else:
        raise ValueError("Unsupported file type")
    
if __name__ == "__main__":
    docs = ingest("data/test.pdf") #input
    print(f"Loaded {len(docs)} documents")

    for d in docs[:2]:
        print("---")
        print(d.metadata)
        print(d.text[:200])