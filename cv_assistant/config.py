"""Настройки читаются из окружения или локального файла .env."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    database_path: Path
    groq_api_key: str
    groq_model: str
    max_followup_questions: int


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    database_path = Path(os.getenv("DATABASE_PATH") or "data/profiles.sqlite3")
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    max_questions = int(os.getenv("MAX_FOLLOWUP_QUESTIONS", "5"))
    if not 1 <= max_questions <= 10:
        raise ValueError("MAX_FOLLOWUP_QUESTIONS должен быть от 1 до 10")
    return Settings(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        database_path=database_path,
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        max_followup_questions=max_questions,
    )

