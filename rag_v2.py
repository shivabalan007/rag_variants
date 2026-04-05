from orchestration.rag_pipeline import RAGPipeline

_pipeline_cache = {}                                        

def run_rag_v2(query, embedder, store, chunks):

    if "pipeline" not in _pipeline_cache:
        _pipeline_cache["pipeline"] = RAGPipeline(embedder, store, chunks)

    pipeline = _pipeline_cache["pipeline"]

    result = pipeline.run(query)

    return result.answer, result.faithfulness, result.relevance 