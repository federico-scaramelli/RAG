from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import settings

# Embedding management utility for the RAG pipeline.
# Loads a SentenceTransformer model (optionally from a local cache),
# generates vector embeddings for input texts, and exposes the
# embedding dimensionality used by the vector store and retriever.

class EmbeddingManager:
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model_name
        self.model: SentenceTransformer | None = None
        self._load_model()

    def _load_model(self) -> None:
        local_path = settings.project_root / f".{self.model_name}"

        if Path(local_path).exists():
            self.model = SentenceTransformer(str(local_path))
        else:
            self.model = SentenceTransformer(self.model_name)

    def generate_embeddings(self, texts: Sequence[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Embedding model not loaded")

        if not texts:
            raise ValueError("texts must not be empty")

        embeddings = self.model.encode(
            list(texts),
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embeddings

    def get_embedding_dimension(self) -> int:
        if self.model is None:
            raise ValueError("Embedding model not loaded")
        return self.model.get_sentence_embedding_dimension()