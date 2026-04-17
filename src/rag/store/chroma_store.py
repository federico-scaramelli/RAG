from __future__ import annotations

from typing import Any, Sequence
from uuid import uuid4

import chromadb
import numpy as np

from rag.config import settings

# Persistent Chroma-based vector store for the RAG pipeline.
# Manages collection initialization, document and embedding insertion,
# similarity queries, collection statistics, and full collection reset
# for ingestion and retrieval workflows.

class VectorStore:
    def __init__(
        self,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_collection_name
        self.persist_directory = str(persist_directory or settings.vectorstore_dir)

        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        settings.ensure_directories()

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings for RAG"},
        )

    def add_documents(self, documents: Sequence[Any], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        if self.collection is None:
            raise ValueError("Collection not initialized")

        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        documents_text: list[str] = []
        embeddings_list: list[list[float]] = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings, strict=False)):
            doc_id = f"doc_{uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=documents_text,
        )

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> dict[str, Any]:
        if self.collection is None:
            raise ValueError("Collection not initialized")

        return self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

    def count(self) -> int:
        if self.collection is None:
            raise ValueError("Collection not initialized")
        return self.collection.count()

    def reset_collection(self) -> None:
        """Reset the collection maintaining path and name"""
        if self.client is None:
            self._initialize_store()

        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # Se non esiste ancora, va bene lo stesso
            pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document embeddings for RAG"},
        )