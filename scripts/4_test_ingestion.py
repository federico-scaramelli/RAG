from __future__ import annotations

from rag.embeddings import EmbeddingManager
from rag.pipeline import IngestPipeline
from rag.store import VectorStore

# End-to-end ingestion test for the RAG pipeline.
# Initializes the embedding manager and vector store, runs the ingestion
# process with collection reset enabled, and prints load, chunking,
# collection size, and embedding dimension statistics.

def main() -> None:
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()

    pipeline = IngestPipeline(
        embedding_manager=embedding_manager,
        vector_store=vector_store,
    )

    result = pipeline.run(
        reset_collection=True,   # così non duplichi
    )

    print(f"Documents loaded: {result.documents_loaded}")
    print(f"Chunks created: {result.chunks_created}")
    print(f"Collection before: {result.collection_count_before}")
    print(f"Collection after: {result.collection_count_after}")
    print(f"Embedding dimension: {result.embedding_dimension}")


if __name__ == "__main__":
    main()