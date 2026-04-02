from dataclasses import dataclass
from langchain_groq import ChatGroq

SYSTEM_PROMPT = """You are a helpful assistant.
Answer using only the provided context.
If the context is insufficient, say so clearly.
Be concise but informative.
"""

USER_PROMPT = """Context:
{context}

Question:
{query}

Answer:"""

@dataclass
class AnswerGenerator:
    llm: ChatGroq

    def generate(self, query: str, context: str) -> str:
        if not context.strip():
            return "No relevant context found to answer the question."

        prompt = f"{SYSTEM_PROMPT}\n\n" + USER_PROMPT.format(
            context=context,
            query=query,
        )
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)