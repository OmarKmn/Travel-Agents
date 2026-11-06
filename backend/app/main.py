"""Point d'entrée de l'API FastAPI."""

from functools import lru_cache

from fastapi import Depends, FastAPI
from pydantic import BaseModel

from app.config import get_settings
from app.llm import LLM, create_llm

app = FastAPI(title="Travel Agents API", version="0.1.0")


@lru_cache
def get_llm() -> LLM:
    return create_llm(get_settings())


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    provider: str


@app.get("/health")
def health() -> dict:
    """Vérifie que l'API tourne et indique le LLM configuré."""
    settings = get_settings()
    return {"status": "ok", "llm_provider": settings.llm_provider, "llm_model": settings.llm_model}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, llm: LLM = Depends(get_llm)) -> ChatResponse:
    """Envoie un message simple au LLM (endpoint de test, avant l'arrivée des agents)."""
    answer = llm.chat([{"role": "user", "content": request.message}])
    return ChatResponse(answer=answer, provider=llm.name)
