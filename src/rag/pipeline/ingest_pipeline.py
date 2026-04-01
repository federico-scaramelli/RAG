from __future__ import annotations

from dataclasses import dataclass

from rag.embeddings import EmbeddingManager
from rag.loaders import load_pdf_documents
from rag.processing.chunking import split_documents
from rag.store import VectorStore
from rag.config import settings


@dataclass
class IngestResult:
    documents_loaded: int
    chunks_created: int
    collection_count_before: int
    collection_count_after: int
    embedding_dimension: int


class IngestPipeline:
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def run(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        reset_collection: bool = True,
    ) -> IngestResult:
        if reset_collection:
            self.vector_store.reset_collection()

        documents = load_pdf_documents()

        if not documents:
            raise ValueError("Nessun documento trovato in data/raw")

        chunks = split_documents(
            documents=documents,
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
        )

        if not chunks:
            raise ValueError("Nessun chunk generato dal chunking")

        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedding_manager.generate_embeddings(texts)

        count_before = self.vector_store.count()
        self.vector_store.add_documents(chunks, embeddings)
        count_after = self.vector_store.count()

        return IngestResult(
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            collection_count_before=count_before,
            collection_count_after=count_after,
            embedding_dimension=self.embedding_manager.get_embedding_dimension(),
        )