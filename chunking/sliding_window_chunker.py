def sliding_window_chunk(text, chunk_size=300, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

"""
Takes a text string and splits into overlapping windows of fixed size. Overlap ensures context at chunk boundaries is not lost between adjacent chunks.
"""