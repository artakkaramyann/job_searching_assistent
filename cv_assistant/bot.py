"""Telegram-интерфейс с кнопками и подтверждением очистки анкеты."""

import asyncio
import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import SimpleEventIsolation
from aiogram.types import BotCommand, MenuButtonCommands, Message

from cv_assistant.interview import Interview
from cv_assistant.telegram_ui import ACTIONS, CANCEL, CONFIRM, keyboard, split_message

logger = logging.getLogger(__name__)


def create_dispatcher(interview: Interview) -> Dispatcher:
    router = Router()
    router.message.filter(F.chat.type == "private")

    @router.message(F.text)
    async def on_text(message: Message, state: FSMContext) -> None:
        if message.from_user is None:
            return
        user_id = message.from_user.id
        text = message.text.strip()
        command = ACTIONS.get(text, text)
        action = command.split()[0].split("@")[0].lower() if command else ""
        view = "dialog"
        try:
            confirmation = await state.get_state() == "confirm_restart"
            mode = await asyncio.to_thread(interview.store.get_ui_mode, user_id)
            if action == "/restart":
                await state.set_state("confirm_restart")
                reply = "🔄 Начать новую анкету? Текущие ответы будут удалены."
                view = "confirm"
            elif text == CONFIRM:
                await state.clear()
                reply = await asyncio.to_thread(interview.handle, user_id, "/restart") if confirmation else "Открой 📋 Меню, чтобы начать заново."
                view = "dialog" if confirmation else "menu"
            elif text == CANCEL:
                await state.clear()
                reply = "✅ Анкета сохранена. Выбери действие."
                view = "menu"
            elif mode == "menu" and text not in ACTIONS and not text.startswith("/"):
                reply = "Неизвестная команда. Выбери кнопку в меню. Для ответа на вопросы нажми «💬 Продолжить»."
                view = "confirm" if confirmation else "menu"
            else:
                await state.clear()
                reply = await asyncio.to_thread(interview.handle, user_id, command)
                if action in {"/menu", "/help", "/profile", "/history"}:
                    view = "menu"
                elif action in {"/review", "/review_full"}:
                    view = "review"
                elif mode == "menu" and action not in {"/start", "/go", "/resume", "/retry", "/continue"}:
                    view = "menu"
            await asyncio.to_thread(interview.store.set_ui_mode, user_id,
                                    "dialog" if view == "dialog" else "menu")
            profile = await asyncio.to_thread(interview.get_state, user_id)
            markup = keyboard(profile, view)
        except Exception as error:
            logger.error("Не удалось обработать сообщение: %s", type(error).__name__)
            await message.answer("⏳ Не удалось обработать ответ. Открой меню и нажми «Продолжить».")
            return
        chunks = split_message(reply)
        for index, chunk in enumerate(chunks):
            await message.answer(chunk, reply_markup=markup if index == len(chunks)-1 else None)

    @router.message()
    async def on_other_message(message: Message) -> None:
        await message.answer("💬 Пока пришли ответ текстом — фото и голос добавим позже.")

    dispatcher = Dispatcher(events_isolation=SimpleEventIsolation())
    dispatcher.include_router(router)
    return dispatcher


async def configure_menu(bot: Bot) -> None:
    await bot.set_my_commands([
        BotCommand(command="menu", description="📋 Открыть меню"),
        BotCommand(command="start", description="🚀 Начать диалог"),
        BotCommand(command="profile", description="👤 Мой профиль"),
    ])
    await bot.set_chat_menu_button(menu_button=MenuButtonCommands())


async def run_bot(token: str, interview: Interview) -> None:
    dispatcher = create_dispatcher(interview)
    async with Bot(token=token) as bot:
        await configure_menu(bot)
        await dispatcher.start_polling(bot, close_bot_session=False)
