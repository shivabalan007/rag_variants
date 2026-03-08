from ingestion.base import Document
from chunking.base import ChunkConfig

def fixed_chunk_document(doc: Document, config: ChunkConfig):
    text = doc.text
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + config.chunk_size
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunk_metadata = doc.metadata.copy()
            chunk_metadata["chunk_id"] = f"{chunk_id}"

            chunks.append(
                Document(
                    text=chunk_text,
                    metadata=chunk_metadata
                )
            )

        start = end - config.overlap
        chunk_id += 1
        
    return chunks