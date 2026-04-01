from rag.embeddings.sentence_transformer import EmbeddingManager
from rag.store.chroma_store import VectorStore
from rag.retrieval.retriever import RAGRetriever
from rag.generation.llm_factory import build_llm
from rag.generation.answerer import AnswerGenerator
from rag.pipeline.rag_pipeline import RAGPipeline

embeddingmanager = EmbeddingManager(model_name="all-MiniLM-L6-v2")
vectorstore = VectorStore(
    collection_name="pdf_documents",
    persist_directory="./data/vectorstore",
    embedding_manager=embeddingmanager,
)
ragretriever = RAGRetriever(
    vector_store=vectorstore,
    embedding_manager=embeddingmanager,
)

llm = build_llm()
answer_generator = AnswerGenerator(llm=llm)

pipeline = RAGPipeline(
    retriever=ragretriever,
    answer_generator=answer_generator,
)