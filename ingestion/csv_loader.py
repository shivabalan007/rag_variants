import pandas as pd 
from ingestion.base import Document

def load_csv(path: str):
    df = pd.read_csv(path)
    documents = []

    for idx, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])

        documents.append(
            Document(
                text=row_text,
                metadata={
                    "source": path,
                    "row": idx,
                    "doc_type": "csv"
                }
            )
        )

    return documents