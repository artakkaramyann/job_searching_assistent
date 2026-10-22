import asyncio
from datetime import datetime, timezone

from aiogram import Bot
from aiogram.types import Chat, Message, Update, User

from cv_assistant.bot import create_dispatcher
from cv_assistant.interview import ABOUT_QUESTION, Interview
from cv_assistant.storage import ProfileStore


def test_telegram_routes_private_text_and_ignores_groups(tmp_path, monkeypatch):
    replies = []

    async def fake_answer(self, text, **kwargs):
        replies.append(text)

    monkeypatch.setattr(Message, "answer", fake_answer)
    interview = Interview(ProfileStore(tmp_path / "profiles.sqlite3"))

    async def scenario():
        dispatcher = create_dispatcher(interview)
        # Фиктивный токен: никакие вызовы Telegram API в тесте не выполняются.
        async with Bot(token="123456:TEST_TOKEN_ONLY") as bot:
            for index, (chat_type, text) in enumerate(
                [("private", "/start"), ("private", "Developer"), ("group", "Secret")],
                start=1,
            ):
                message = Message(
                    message_id=index,
                    date=datetime.now(timezone.utc),
                    chat=Chat(id=1 if chat_type == "private" else -10, type=chat_type),
                    from_user=User(id=1, is_bot=False, first_name="Test"),
                    text=text,
                )
                await dispatcher.feed_update(bot, Update(update_id=index, message=message))
        await dispatcher.storage.close()
        await dispatcher.fsm.events_isolation.close()

    asyncio.run(scenario())
    assert len(replies) == 2
    assert ABOUT_QUESTION in replies[-1]
    assert interview.store.get(1).target_role == "Developer"
    assert interview.store.get(1).about == ""
