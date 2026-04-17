from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import settings

# Document chunking utility for the RAG pipeline.
# Uses a recursive character-based text splitter to break documents into
# retrieval-friendly chunks, tagging each chunk with its index for
# downstream tracing and debugging.

DEFAULT_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    " ",
    "",
]


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    separators: list[str] | None = None,
) -> list[Document]:
    """Split documents into smaller chunks for retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=separators or DEFAULT_SEPARATORS,
    )

    split_docs = splitter.split_documents(documents)

    for index, doc in enumerate(split_docs):
        doc.metadata["chunk_index"] = index

    return split_docs