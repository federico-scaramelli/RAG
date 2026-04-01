from rag.config import settings


def main() -> None:
    print("project_root:", settings.project_root)
    print("data_dir:", settings.data_dir)
    print("raw_data_dir:", settings.raw_data_dir)
    print("processed_data_dir:", settings.processed_data_dir)
    print("vectorstore_dir:", settings.vectorstore_dir)


if __name__ == "__main__":
    main()