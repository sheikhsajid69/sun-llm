"""Centralized application configuration.

Loads runtime settings from environment variables and optional .env file.
Ensures zero hardcoded credentials and safe defaults.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Base directory for the repository root
BASE_DIR = Path(__file__).resolve().parent.parent

# Attempt to load .env from repository root if it exists
load_dotenv(dotenv_path=BASE_DIR / ".env")


def _str_to_bool(value: Optional[str], default: bool = False) -> bool:
    """Parse string representation of boolean safely."""
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes", "on", "t")


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration container."""

    # Provider Selection
    use_hf: bool = field(
        default_factory=lambda: _str_to_bool(os.getenv("USE_HF"), default=False)
    )

    # Ollama Configuration
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("OLLAMA_MODEL", "mistral")
    )

    # Hugging Face Configuration
    hf_token: str = field(
        default_factory=lambda: os.getenv("HF_TOKEN", "")
    )
    hf_model: str = field(
        default_factory=lambda: os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    )
    hf_base_url: str = field(
        default_factory=lambda: os.getenv("HF_BASE_URL", "https://api-inference.huggingface.co/models").rstrip("/")
    )

    # Memory & Execution Limits
    max_history_messages: int = field(
        default_factory=lambda: int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
    )
    llm_timeout: float = field(
        default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "120.0"))
    )

    # Server Settings
    host: str = field(
        default_factory=lambda: os.getenv("HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: int(os.getenv("PORT", "8000"))
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )

    # Filesystem Paths
    base_dir: Path = BASE_DIR
    templates_dir: Path = BASE_DIR / "templates"
    system_prompt_path: Path = BASE_DIR / "app" / "prompts" / "system.txt"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieve cached application settings instance."""
    return Settings()
