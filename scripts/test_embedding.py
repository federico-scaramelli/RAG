from rag.embeddings import EmbeddingManager

def test_embedding_manager_init():
    manager = EmbeddingManager()
    assert manager.get_embedding_dimension() == 384