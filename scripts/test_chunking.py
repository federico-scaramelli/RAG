from langchain_core.documents import Document
from rag.processing import split_documents

docs = [
    Document(
        page_content="Questo è un test. " * 300,
        metadata={"source": "demo.txt"},
    )
]

chunks = split_documents(docs)

print(f"Numero chunks: {len(chunks)}")
print(chunks[0].metadata)
print(chunks[0].page_content[:200])