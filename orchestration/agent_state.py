class AgentState:
    def __init__(self, query: str):
        #User input
        self.original_query = query
        self.rewritten_query = None

        #Retrieval
        self.similarity_scores = []
        self.candidate_chunks = []
        self.reranked_chunks = []

        #Generation
        self.answer = None

        #Evaluation
        self.overlap = None
        self.faithfulness = None
        self.relevance = None
        
        #Control
        self.attempt = 0
        self.decision = None