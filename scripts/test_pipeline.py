from rag.embeddings import EmbeddingManager
from rag.store import VectorStore
from rag.retrieval import RAGRetriever

em = EmbeddingManager()
vs = VectorStore()
rr = RAGRetriever(vector_store=vs, embedding_manager=em)
print(rr.retrieve("what is HIV?", top_k=5, score_threshold=0.1))