from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from rag.embeddings import EmbeddingManager
from rag.store import VectorStore
from rag.retrieval import RAGRetriever
from rag.generation.answerer import AnswerGenerator
from rag.generation.llm_factory import build_llm
from rag.pipeline import RAGPipeline

# Command-line entry point for the RAG system.
# Builds the full retrieval and answer-generation pipeline,
# accepts a user query, and prints the final answer,
# confidence score, and supporting sources.

def build_pipeline() -> RAGPipeline:
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    retriever = RAGRetriever(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
    )

    llm = build_llm()
    answer_generator = AnswerGenerator(llm=llm)

    return RAGPipeline(
        retriever=retriever,
        answer_generator=answer_generator,
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/ask.py \"your question here\"")
        sys.exit(1)

    query = sys.argv[1]
    pipeline = build_pipeline()

    result = pipeline.answer(
        query=query,
        top_k=5,
        min_score=0.1,
        return_context=True,
    )

    print("\n=== ANSWER ===\n")
    print(result["answer"])
    print("\n=== CONFIDENCE ===\n")
    print(result["confidence"])
    print("\n=== SOURCES ===\n")
    for s in result["sources"]:
        print(
            f"- {s['source']} (page {s['page']}, score {s['score']:.3f})"
        )


if __name__ == "__main__":
    main()