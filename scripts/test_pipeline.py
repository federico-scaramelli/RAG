from rag.embeddings import EmbeddingManager
from rag.store import VectorStore
from rag.retrieval import RAGRetriever

def main() -> None:
    em = EmbeddingManager()
    vs = VectorStore()
    retriever = RAGRetriever(vector_store=vs, embedding_manager=em)

    results = retriever.retrieve("what is HIV?", top_k=5, score_threshold=0.1)
    for r in results:
        print("-" * 80)
        print("ID      :", r["id"])
        print("Score   :", r["similarity_score"])
        print("Source  :", r["metadata"].get("source"))
        print("Page    :", r["metadata"].get("page"))
        print("Preview :", r["content"][:200].replace("\n", " ") + "...")

if __name__ == "__main__":
    main()