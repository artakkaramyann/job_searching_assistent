"""Telegram — только интерфейс: принимает текст и показывает ответ диалога."""

import asyncio
import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.fsm.storage.memory import SimpleEventIsolation
from aiogram.types import Message

from cv_assistant.interview import Interview

logger = logging.getLogger(__name__)


def create_dispatcher(interview: Interview) -> Dispatcher:
    router = Router()
    # Анкеты обрабатываются только в личных сообщениях с ботом.
    router.message.filter(F.chat.type == "private")

    @router.message(F.text)
    async def on_text(message: Message) -> None:
        if message.from_user is None:
            return
        try:
            reply = await asyncio.to_thread(
                interview.handle, message.from_user.id, message.text
            )
        except Exception:
            logger.exception("Не удалось обработать сообщение")
            await message.answer("Не удалось сохранить ответ. Попробуй отправить его ещё раз.")
            return
        # Без HTML-разметки: пользовательский текст показываем как обычный текст.
        # 1500 символов оставляют запас для эмодзи и лимита Telegram.
        for offset in range(0, len(reply), 1500):
            await message.answer(reply[offset : offset + 1500])

    @router.message()
    async def on_other_message(message: Message) -> None:
        await message.answer(
            "Пока я принимаю только текст. Фото для резюме добавим на следующем этапе."
        )

    # Сообщения одного пользователя обрабатываются по очереди.
    dispatcher = Dispatcher(events_isolation=SimpleEventIsolation())
    dispatcher.include_router(router)
    return dispatcher


async def run_bot(token: str, interview: Interview) -> None:
    dispatcher = create_dispatcher(interview)
    async with Bot(token=token) as bot:
        await dispatcher.start_polling(bot, close_bot_session=False)

