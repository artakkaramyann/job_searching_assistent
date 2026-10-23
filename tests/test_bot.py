import asyncio
from datetime import datetime, timezone

from aiogram import Bot
from aiogram.types import Chat, Message, Update, User

from cv_assistant.bot import create_dispatcher
from cv_assistant.interview import ABOUT_QUESTION, Interview
from cv_assistant.llm import OfflineInterviewer
from cv_assistant.storage import ProfileStore
from cv_assistant.telegram_ui import CONFIRM, CANCEL


def test_telegram_routes_private_text_and_ignores_groups(tmp_path, monkeypatch):
    replies = []

    async def fake_answer(self, text, **kwargs):
        replies.append(text)

    monkeypatch.setattr(Message, "answer", fake_answer)
    interview = Interview(ProfileStore(tmp_path / "profiles.sqlite3"), OfflineInterviewer())

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
    assert interview.get_state(1)["target_role"] == "Developer"
    assert interview.get_state(1)["about"] == ""


def test_menu_buttons_and_confirmed_restart(tmp_path, monkeypatch):
    replies = []

    async def fake_answer(self, text, **kwargs):
        replies.append((text, kwargs.get("reply_markup")))

    monkeypatch.setattr(Message, "answer", fake_answer)
    interview = Interview(ProfileStore(tmp_path / "db.sqlite3"), OfflineInterviewer())

    async def scenario():
        dispatcher = create_dispatcher(interview)
        async with Bot(token="123456:TEST_TOKEN_ONLY") as bot:
            async def send(text):
                message = Message(message_id=len(replies)+1, date=datetime.now(timezone.utc),
                                  chat=Chat(id=1, type="private"),
                                  from_user=User(id=1, is_bot=False, first_name="Test"), text=text)
                await dispatcher.feed_update(bot, Update(update_id=len(replies)+1, message=message))

            await send("📋 Меню")
            buttons = {b.text for row in replies[-1][1].keyboard for b in row}
            assert {"🚀 Начать", "👤 Профиль", "🔄 Начать заново"} <= buttons
            await send("🚀 Начать")
            assert interview.get_state(1)["target_role"] == ""
            await send("Повар")
            await send("Готовлю шаурму")
            original = interview.get_state(1)["turns"]
            for action in ["📋 Меню", "👤 Профиль", "💬 Продолжить", CONFIRM]:
                await send(action)
                assert interview.get_state(1)["turns"] == original
            await send("📋 Меню")
            for text in ["случайное сообщение", "/неизвестная", "ещё одно сообщение", "👤 Профиль", "это не ответ"]:
                await send(text)
                assert interview.get_state(1)["turns"] == original
            assert "Неизвестная команда" in replies[-1][0]
            assert interview.store.get_ui_mode(1) == "menu"
            # Состояние экрана сохраняется и при перезапуске процесса/диспетчера.
            await dispatcher.storage.close()
            await dispatcher.fsm.events_isolation.close()
            dispatcher = create_dispatcher(Interview(ProfileStore(interview.store.path), OfflineInterviewer()))
            await send("случайный текст после перезапуска")
            assert "Неизвестная команда" in replies[-1][0]
            assert interview.get_state(1)["turns"] == original
            await send("💬 Продолжить")
            await send("Опыта нет")
            assert len(interview.get_state(1)["turns"]) == len(original) + 1
            original = interview.get_state(1)["turns"]
            await send("🔄 Начать заново")
            assert interview.get_state(1)["turns"] == original
            assert {b.text for row in replies[-1][1].keyboard for b in row} == {CONFIRM, CANCEL}
            await send(CANCEL)
            assert interview.get_state(1)["turns"] == original
            await send("/restart")
            await send(CONFIRM)
            assert interview.get_state(1)["turns"] == []
            assert interview.get_state(1)["target_role"] == ""
        await dispatcher.storage.close()
        await dispatcher.fsm.events_isolation.close()

    asyncio.run(scenario())


def test_native_menu_configuration():
    from unittest.mock import AsyncMock
    from cv_assistant.bot import configure_menu
    bot = AsyncMock()
    asyncio.run(configure_menu(bot))
    assert [c.command for c in bot.set_my_commands.call_args.args[0]] == ["menu", "start", "profile"]
    assert bot.set_chat_menu_button.call_args.kwargs["menu_button"].type == "commands"
