from dataclasses import dataclass
from typing import Any

# High-level RAG orchestration pipeline.
# Connects a retriever and an answer generator to handle user queries end-to-end:
# fetches relevant documents, builds a context string, and returns the final answer,
# confidence score, and structured source metadata.

@dataclass
class RAGPipeline:
    retriever: Any
    answer_generator: Any

    def answer(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        return_context: bool = False,
    ) -> dict:
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=min_score,
        )

        if not results:
            return {
                "answer": "No relevant context found.",
                "sources": [],
                "confidence": 0.0,
                "context": "" if return_context else None,
            }

        context = "\n\n".join(doc["content"] for doc in results)

        sources = [
            {
                "source": doc["metadata"].get("source", "unknown"),
                "page": doc["metadata"].get("page", "unknown"),
                "score": doc["similarity_score"],
                "preview": doc["content"][:300] + "...",
            }
            for doc in results
        ]

        confidence = max(doc["similarity_score"] for doc in results)
        answer = self.answer_generator.generate(query=query, context=context)

        output = {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
        }

        if return_context:
            output["context"] = context

        return output