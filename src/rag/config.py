from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT

    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    vectorstore_dir: Path = data_dir / "vectorstore"

    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "pdfdocuments")

    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "openai/gpt-oss-120b")

    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    default_min_score: float = float(os.getenv("DEFAULT_MIN_SCORE", "0.1"))

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()