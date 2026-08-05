from llm.openrouter_client import llm


class IntentRouter:

    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    GENERAL = "general"

    def classify(self, query: str, conversation_history=None) -> str:

        history_text = ""

        if conversation_history:
            for message in conversation_history:
                history_text += (
                    f"{message['role'].capitalize()}: "
                    f"{message['content']}\n"
                )

        prompt = f"""
You are an expert Intent Classification System.

Your task is to classify EVERY user query into EXACTLY ONE intent.

==================================================
INTENT DEFINITIONS
==================================================

1. memory

Use this intent ONLY when the user is referring to
previous conversation, stored information, previous
answers, personal facts shared earlier, or conversation
history.

Examples:

- What is my name?
- What did I ask earlier?
- What was your previous answer?
- Summarize our conversation.
- Do you remember what I said?
- What did I just tell you?
- Repeat your previous response.
- What was my first question?
- Continue from where we stopped.
- Did I mention my favourite language?
- Remind me what we discussed.
- Who am I?
- What are my goals?
- What did we discuss yesterday?


==================================================

2. knowledge

Use this intent when the user wants factual,
technical, educational, coding, document,
or web-based information.

Examples:

- What is Python?
- Explain OOP.
- Explain Retrieval Augmented Generation.
- Explain Transformers.
- Explain this error.
- Debug this code.
- Compare SQL and NoSQL.
- Write Python code.
- Summarize this document.
- What is written in the uploaded PDF?
- Explain this topic.
- How does BM25 work?
- What happened in FIFA 2026?
- Difference between CNN and RNN.
- According to the uploaded file...
- Explain this research paper.


==================================================

3. general

Use this intent for greetings,
casual conversation,
small talk,
gratitude,
or questions that do NOT require
memory or document retrieval.

Examples:

- Hi
- Hello
- Good morning
- Good evening
- Bye
- Thank you
- Thanks
- Nice to meet you
- Tell me a joke
- How are you?
- Who are you?
- Can you help me?


==================================================
Conversation History
==================================================

{history_text if history_text else "No previous conversation."}

==================================================
Current User Query
==================================================

{query}

==================================================
Decision Rules
==================================================

1. Return EXACTLY ONE intent.

2. If the question depends on previous conversation,
choose:

memory

3. If the question requires factual knowledge,
documents,
uploaded files,
coding,
technical explanation,
or web knowledge,
choose:

knowledge

4. Greetings,
casual chat,
introductions,
and friendly conversation
should be:

general

5. If the query could belong to BOTH memory
and knowledge,
choose the PRIMARY purpose.

Example:

"What did we discuss about Python?"

Primary purpose:
Retrieve previous conversation.

Return:

memory

Example:

"Explain Python."

Return:

knowledge

6. If unsure,
prefer:

knowledge


==================================================
Output Rules
==================================================

Return ONLY ONE WORD.

Allowed outputs:

memory

knowledge

general

Do NOT explain.

Do NOT add punctuation.

Do NOT add extra words.

"""

        response = llm(
            prompt=prompt,
            temperature=0.0
        )

        if response is None:
            return self.KNOWLEDGE

        intent = response.strip().lower()

        if intent == self.MEMORY:
            return self.MEMORY

        if intent == self.GENERAL:
            return self.GENERAL

        if intent == self.KNOWLEDGE:
            return self.KNOWLEDGE

        return self.KNOWLEDGE

"""
    Classifies the user's query into one of three intents - memory, knowledge, general
"""