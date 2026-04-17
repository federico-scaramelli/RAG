from rag.embeddings import EmbeddingManager

# Unit test for the embedding manager initialization.
# Verifies that the configured embedding model returns the expected
# vector dimensionality used by the vector store and retrieval pipeline.

def test_embedding_manager_init():
    manager = EmbeddingManager()
    assert manager.get_embedding_dimension() == 384