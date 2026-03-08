from ingestion.run_ingestion import ingest
from chunking.base import ChunkConfig
from chunking.fixed_chunker import fixed_chunk_document

from embeddings.base import EmbeddingConfig
from embeddings.embedder import Embedder

from retrieval.vector_store import VectorStore
from retrieval.query_rewriter import rewrite_query
from retrieval.reranker import CrossEncoderReranker

from llm.generator import generate_answer

from evaluation.faithfulness import check_faithfulness
from evaluation.relevance import check_relevance


def main():

    # INGEST DOCUMENTS
    docs = ingest("data/test1.txt")


    # CHUNK DOCUMENTS
    chunk_config = ChunkConfig(
        chunk_size=300,
        overlap=50
    )

    chunks = []

    for doc in docs:
        chunks.extend(
            fixed_chunk_document(doc, chunk_config)
        )

    print(f"Total chunks created: {len(chunks)}")

    # LOAD EMBEDDING MODEL
    embedder = Embedder(
        EmbeddingConfig(
            model_name="all-MiniLM-L6-v2"
        )
    )

    # CREATE EMBEDDINGS
    embeddings = embedder.embed_documents(chunks)

    # BUILD VECTOR STORE
    dim = embeddings.shape[1]

    store = VectorStore(dim)

    store.add(embeddings)

    print(f"Stored {store.index.ntotal} vectors")

    # USER QUERY
    query = input("Enter your question: ")

    # RETRY LOOP
    for k in [3, 5]:

        print(f"\n=== RETRIEVAL WITH k={k} ===\n")

        # QUERY REWRITE
        rewritten_query = rewrite_query(query)

        print("Original Query:", query)
        print("Rewritten Query:", rewritten_query)

        # QUERY EMBEDDING
        query_vector = embedder.embed_query(rewritten_query)

        # VECTOR SEARCH
        scores, indices = store.search(
            query_vector,
            top_k=k
        )

        # EXTRACT CHUNK TEXT
        candidate_chunks = [
            chunks[idx] for idx in indices[0]
        ]

        # RERANK RESULTS
        reranker = CrossEncoderReranker()

        reranked_chunks = reranker.rerank(
            rewritten_query,
            candidate_chunks,
            top_k=3
        )

        retrieved_texts = [
            chunk.text for chunk in reranked_chunks
        ]

        # GENERATE ANSWER
        answer = generate_answer(
            query,
            retrieved_texts
        )

        # EVALUATE ANSWER
        verdict = check_faithfulness(
            query,
            answer,
            retrieved_texts
        )

        relevance = check_relevance(
            query,
            answer
        )

        # RESULTS
        print("\nANSWER:\n", answer)

        print("\nFAITHFULNESS:", verdict)

        print("\nRELEVANCE:", relevance)

        # DECISION LOGIC
        if verdict.upper() == "YES" and relevance.upper() == "YES":

            print("\nFINAL DECISION: ACCEPTED")

            break

    else:

        answer = "I don't know based on the provided context."

        print("\nFINAL DECISION: REFUSED")


if __name__ == "__main__":
    main()



