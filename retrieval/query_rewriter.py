from llm.openrouter_client import llm


def rewrite_query(query, conversation_history=None):

    history_text = ""

    if conversation_history:
        for msg in conversation_history:
            history_text += (
                f"{msg['role'].capitalize()}: "
                f"{msg['content']}\n"
            )

    prompt = f"""
You are an expert Query Rewriting Assistant for a Retrieval-Augmented Generation (RAG) system.

Your goal is to improve retrieval quality by rewriting the user's query into a clear, standalone query while preserving the original intent.

==================================================
Conversation History
==================================================

{history_text if history_text else "No previous conversation."}

==================================================
Current User Query
==================================================

{query}

==================================================
Responsibilities
==================================================

Rewrite the query ONLY when doing so improves semantic retrieval.

Your rewritten query should be:

- self-contained
- grammatically correct
- natural English
- optimized for vector search and keyword search

==================================================
Rewrite Guidelines
==================================================

1. Preserve the user's original meaning exactly.

2. Use conversation history ONLY to resolve references such as:

- it
- this
- that
- they
- them
- previous answer
- earlier discussion
- above
- former
- latter

3. Expand abbreviations when appropriate.

Examples:

AI
→ Artificial Intelligence

RAG
→ Retrieval-Augmented Generation

OOP
→ Object-Oriented Programming

NLP
→ Natural Language Processing

4. Preserve technical terms,
variable names,
class names,
function names,
API names,
library names,
file names,
error messages,
and code snippets exactly.

5. Do NOT answer the question.

6. Do NOT summarize.

7. Do NOT explain.

8. Do NOT infer missing facts.

9. Do NOT introduce new information.

10. Do NOT change the intent.

11. If the original query is already clear and self-contained,
return it unchanged.

==================================================
Examples
==================================================

Conversation:

User: Explain Python.

Current Query:

How does it handle memory?

Rewrite:

How does Python handle memory?


Conversation:

User: Explain OOP.

Current Query:

What are its advantages?

Rewrite:

What are the advantages of Object-Oriented Programming (OOP)?


Conversation:

Current Query:

What is BM25?

Rewrite:

What is BM25?


Conversation:

Current Query:

Debug this code.

Rewrite:

Debug this code.


Conversation:

Current Query:

Compare SQL and NoSQL.

Rewrite:

Compare SQL and NoSQL.


==================================================
Output Rules
==================================================

Return ONLY the rewritten query.

Do NOT include explanations.

Do NOT include notes.

Do NOT include quotation marks.

Return exactly one sentence.

"""

    rewritten = llm(
        prompt,
        temperature=0.0
    )

    rewritten = " ".join(rewritten.split())

    return rewritten.strip()

"""
Takes raw user query and rewrites it to be more specific and retrieval-friendly. Expands abbreviations, adds context — improves chunk matching quality.
"""
