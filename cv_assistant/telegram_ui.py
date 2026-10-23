"""Кнопки Telegram: пользователю не нужно помнить команды."""

from aiogram.types import KeyboardButton, ReplyKeyboardMarkup

ACTIONS = {
    "🚀 Начать": "/go", "💬 Продолжить": "/go", "👤 Профиль": "/profile",
    "📝 Рекомендации": "/review", "📖 Подробнее": "/review_full",
    "✅ Завершить опрос": "/continue", "📋 Меню": "/menu",
    "🔄 Начать заново": "/restart",
}
CONFIRM = "🗑 Да, очистить"
CANCEL = "↩️ Отмена"


def keyboard(state: dict, view: str = "dialog") -> ReplyKeyboardMarkup:
    if view == "confirm":
        rows = [[CONFIRM, CANCEL]]
    elif view == "review":
        rows = [["📖 Подробнее", "💬 Продолжить"], ["👤 Профиль", "📋 Меню"]]
    elif view == "menu":
        rows = [["💬 Продолжить" if state["target_role"] else "🚀 Начать", "👤 Профиль"],
                ["📝 Рекомендации", "🔄 Начать заново"], ["📋 Меню"]]
    elif state["stage"] == "complete":
        rows = [["👤 Профиль", "📝 Рекомендации"], ["💬 Продолжить", "📋 Меню"]]
    else:
        rows = [["💬 Продолжить", "✅ Завершить опрос"], ["👤 Профиль", "📋 Меню"]]
    return ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text=text) for text in row] for row in rows],
                               resize_keyboard=True, is_persistent=True,
                               input_field_placeholder="Напиши ответ или выбери действие…")


def split_message(text: str, limit: int = 1800) -> list[str]:
    """Делим длинный разбор по абзацам/словам, а не посередине слова."""
    chunks = []
    while len(text) > limit:
        end = text.rfind("\n\n", 0, limit + 1)
        if end < limit // 2:
            end = text.rfind("\n", 0, limit + 1)
        if end < limit // 2:
            end = text.rfind(" ", 0, limit + 1)
        if end <= 0:
            end = limit
        chunks.append(text[:end])
        text = text[end:].lstrip()
    if text:
        chunks.append(text)
    return chunks
