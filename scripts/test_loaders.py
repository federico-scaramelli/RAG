from rag.config import settings
from rag.loaders import load_all_supported_documents, load_pdf_documents, load_text_documents


def main() -> None:
    print("Using raw data dir:", settings.raw_data_dir)

    text_docs = load_text_documents()
    pdf_docs = load_pdf_documents()
    all_docs = load_all_supported_documents()

    print(f"text docs: {len(text_docs)}")
    print(f"pdf docs: {len(pdf_docs)}")
    print(f"all docs: {len(all_docs)}")

    if all_docs:
        print("first metadata:", all_docs[0].metadata)
        print("first preview:", all_docs[0].page_content[:300])


if __name__ == "__main__":
    main()