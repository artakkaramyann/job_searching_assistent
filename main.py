"""Точка входа: python main.py или python main.py --demo."""

import argparse
import asyncio
import logging

from cv_assistant.config import load_settings
from cv_assistant.interview import Interview
from cv_assistant.llm import GroqInterviewer, OfflineInterviewer
from cv_assistant.reviewer import GroqReviewer, OfflineReviewer
from cv_assistant.storage import ProfileStore


def run_demo(interview: Interview) -> None:
    # Отрицательный ID отделяет локальную анкету от Telegram-пользователей.
    user_id = -1
    print("Локальный диалог. Для выхода введи /exit.\n")
    print("Бот:", interview.handle(user_id, "/start"))
    while True:
        try:
            text = input("\nТы: ")
        except (EOFError, KeyboardInterrupt):
            print("\nДиалог закрыт. Ответы сохранены.")
            break
        if text.strip().lower() == "/exit":
            break
        print("\nБот:", interview.handle(user_id, text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Помощник для подготовки резюме")
    parser.add_argument("--demo", action="store_true", help="Диалог без Telegram и токена")
    parser.add_argument("--offline", action="store_true", help="Учебный режим без LLM и API-ключа")
    args = parser.parse_args()
    try:
        settings = load_settings()
    except ValueError as error:
        parser.exit(status=1, message=f"Ошибка настройки: {error}\n")
    if not args.demo and not settings.telegram_bot_token:
        parser.exit(
            status=1,
            message=(
                "Добавь TELEGRAM_BOT_TOKEN в файл .env (пример в .env.example).\n"
                "Без токенов: python main.py --demo --offline\n"
            ),
        )
    if not args.offline and not settings.groq_api_key:
        parser.exit(status=1, message=(
            "Добавь GROQ_API_KEY в .env. Модель: openai/gpt-oss-120b.\n"
            "Без API: python main.py --demo --offline\n"
        ))
    analyzer = (OfflineInterviewer() if args.offline else
                GroqInterviewer(settings.groq_api_key, settings.groq_model))
    reviewer = (OfflineReviewer() if args.offline else
                GroqReviewer(settings.groq_api_key, settings.groq_model))
    interview = Interview(ProfileStore(settings.database_path), analyzer,
                          settings.max_followup_questions, reviewer)
    print("Режим: " + ("OFFLINE — фиксированные вопросы, без ИИ" if args.offline
                       else f"Groq / {settings.groq_model}"))
    if args.demo:
        run_demo(interview)
    else:
        from cv_assistant.bot import run_bot

        logging.basicConfig(level=logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        asyncio.run(run_bot(settings.telegram_bot_token, interview))


if __name__ == "__main__":
    main()

