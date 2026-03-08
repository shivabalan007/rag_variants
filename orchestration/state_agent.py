class AgentState:
    def __init__(self, query):
        self.query = query
        self.retrieved_chunk = None
        self.answer = None
        self.overlap = None
        self.faithfulness = None
        self.attempt = 0

class AgenticRAG:
    def __init__(self, retriever, generator, evaluator, decision_engine):
        self.retriever = retriever 
        self.generator = generator
        self.evaluator = evaluator
        self.decision_engine = decision_engine

    def run(self, query, max_attempts=2):
        state = AgentState(query)

        while state.attempt < max_attempts:
            state.attempt += 1

            #Retrieve
            state.retrieved_chunk = self.retriever(query, top_k=3 + state.attempt)

            #Generate
            state.answer = self.generator.generate(query, state.retrieved_chunk)

            #Evaluate
            state.overlap = self.evaluator.overlap(state.answer, state.retrieved_chunk)
            state.faithfulness = self.evaluator.faithfulness(state.answer, state.retrieved_chunk)

            #Decide
            decision = self.decision_engine.decide(state.overlap, state.faithfulness)

            print(f"\n Attempt {state.attempt}")
            print("Overlap:", state.overlap)
            print("Faithfulness:", state.faithfulness)
            print("Decision:", decision)

            if decision == "ACCEPT":
                return state.answer
            
        return "I don't know based on the provided context."