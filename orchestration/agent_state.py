class AgentState:
    def __init__(self, query: str):

        #User input
        self.original_query = query
        self.rewritten_query = None

        #Retrieval
        self.retrieval_result = None

        #Routing
        self.route = None
        self.route_reason = None
        self.route_confidence = None
        self.confidence_level = None

        #Web
        self.web_result = None

        # Final answer source
        self.answer_source = None

        # Warning
        self.warning = None
        
        #Generation
        self.answer = None

        #Evaluation
        self.overlap = None
        self.faithfulness = None
        self.relevance = None
        
        #Control
        self.attempt = 0
        self.decision = None

"""
Defines AgentState dataclass that holds all pipeline state — query, chunks, answer, scores, decision, attempt number. Passed between every stage in RAG v2.
"""