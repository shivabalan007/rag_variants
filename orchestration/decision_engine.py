class DecisionEngine:
    def __init__(self,overlap_threshold=0.3):
        self.overlap_threshold = overlap_threshold

    def decide(self, overlap_score, faithfulness_result, relevance_result):
        if overlap_score < self.overlap_threshold:
            return "RETRY"
        
        if faithfulness_result.strip().upper() != "YES":
            return "RETRY"

        if relevance_result.strip().upper() != "YES":
            return "RETRY"
            
        return "ACCEPT"
    
"""
Takes overlap score, faithfulness, and relevance — returns ACCEPT or RETRY. Three-signal decision logic — all three must pass for answer to be accepted.
"""