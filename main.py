"""Точка входа: python main.py или python main.py --demo."""

import argparse
import asyncio
import logging

from cv_assistant.config import load_settings
from cv_assistant.interview import Interview
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
    args = parser.parse_args()
    settings = load_settings()
    if not args.demo and not settings.telegram_bot_token:
        parser.exit(
            status=1,
            message=(
                "Добавь TELEGRAM_BOT_TOKEN в файл .env (пример в .env.example).\n"
                "Без токена можно запустить: python main.py --demo\n"
            ),
        )
    interview = Interview(ProfileStore(settings.database_path))
    if args.demo:
        run_demo(interview)
    else:
        from cv_assistant.bot import run_bot

        logging.basicConfig(level=logging.INFO)
        asyncio.run(run_bot(settings.telegram_bot_token, interview))


if __name__ == "__main__":
    main()

