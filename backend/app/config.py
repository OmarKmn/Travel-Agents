"""Configuration de l'application, lue depuis les variables d'environnement / le fichier .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Dossiers des données (P-Project/data), ignorés par Git
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"  # pages Wikivoyage téléchargées
PROCESSED_DIR = DATA_DIR / "processed"  # chunks et lieux
INDEX_DIR = DATA_DIR / "index"  # index de recherche


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["fake", "ollama"] = "fake"
    llm_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"

    embedding_provider: Literal["fake", "ollama"] = "fake"
    embedding_model: str = "bge-m3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
