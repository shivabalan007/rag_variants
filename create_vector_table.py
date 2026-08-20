from database.connection import engine, Base
from database.vector_models import DocumentChunk

Base.metadata.create_all(bind=engine)

print("document_chunks table created successfully.")