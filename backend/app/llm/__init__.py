from app.config import Settings
from app.llm.base import LLM, Message
from app.llm.fake import FakeLLM
from app.llm.ollama import OllamaLLM

__all__ = ["LLM", "Message", "FakeLLM", "OllamaLLM", "create_llm"]


def create_llm(settings: Settings) -> LLM:
    """Crée le LLM choisi dans la configuration (LLM_PROVIDER)."""
    if settings.llm_provider == "ollama":
        return OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)
    return FakeLLM()
