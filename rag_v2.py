from orchestration.rag_pipeline import RAGPipeline


def run_rag_v2(query, embedder, store, chunks):

    pipeline = RAGPipeline(embedder, store, chunks)

    result = pipeline.run(query)

    return result.answer, result.faithfulness, result.relevance