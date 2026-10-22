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


def load_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    database_path = Path(os.getenv("DATABASE_PATH") or "data/profiles.sqlite3")
    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path
    return Settings(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        database_path=database_path,
    )

