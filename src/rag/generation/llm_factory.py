from dotenv import load_dotenv
from langchain_groq import ChatGroq

from rag.config import settings

load_dotenv()

def build_llm() -> ChatGroq:
    api_key = settings.groq_api_key
    model_name = settings.groq_model_name

    if not api_key:
        raise RuntimeError("GROQ_API_KEY non trovato nell'ambiente (.env).")

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.1,
        max_tokens=1024,
    )