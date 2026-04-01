from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader, TextLoader

from rag.config import settings


def load_text_documents(directory: str | Path | None = None) -> list[Any]:
    target_dir = Path(directory) if directory else settings.raw_data_dir

    loader = DirectoryLoader(
        str(target_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    return loader.load()


def load_pdf_documents(directory: str | Path | None = None) -> list[Any]:
    target_dir = Path(directory) if directory else settings.raw_data_dir

    loader = DirectoryLoader(
        str(target_dir),
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=False,
    )
    return loader.load()


def load_all_supported_documents(directory: str | Path | None = None) -> list[Any]:
    docs: list[Any] = []
    docs.extend(load_text_documents(directory))
    docs.extend(load_pdf_documents(directory))
    return docs