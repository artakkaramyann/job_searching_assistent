"""Логика первого диалога. Она работает и в Telegram, и в терминале.

Пока используются заранее заданные вопросы. На следующем этапе LangGraph
и языковая модель будут выбирать уточнения по содержанию ответов.
"""

from cv_assistant.storage import Profile, ProfileStore

ROLE_QUESTION = "На какую должность ты хочешь откликаться? Например: Junior Python Developer."
ABOUT_QUESTION = (
    "Расскажи о себе одним сообщением: чем занимался, какие есть навыки, "
    "проекты и образование. Учебные проекты тоже подходят."
)
COMPLETED = (
    "Опрос завершён. Данные сохранены. Посмотреть их: /profile.\n"
    "Это первый этап проекта: ИИ-анализ и создание PDF добавим позже."
)
HELP = (
    "Это учебная версия помощника для резюме. Ответы сохраняются локально в SQLite.\n\n"
    "/start — начать или продолжить опрос\n"
    "/profile — посмотреть сохранённые ответы\n"
    "/continue — завершить опрос с текущими данными\n"
    "/restart — очистить текущую анкету и начать заново\n"
    "/help — показать команды"
)


def profile_text(profile: Profile) -> str:
    return (
        f"Желаемая должность: {profile.target_role or 'пока не указана'}\n\n"
        f"О себе:\n{profile.about or 'пока нет информации'}"
    )


class Interview:
    def __init__(self, store: ProfileStore):
        self.store = store

    def handle(self, user_id: int, text: str) -> str:
        """Обрабатываем одно сообщение, затем сохраняем новое состояние анкеты."""
        text = text.strip()
        if not text:
            return "Пришли, пожалуйста, текстовый ответ."
        if len(text) > 8000:
            return "Сократи ответ до 8000 символов и отправь снова."

        profile = self.store.get(user_id)
        # /start@bot_name тоже является командой /start в Telegram.
        command = text.split()[0].split("@")[0].lower()
        if command == "/help":
            return HELP
        if command == "/profile":
            return profile_text(profile)
        if command == "/restart":
            self.store.save(Profile(user_id=user_id))
            return "Текущая анкета очищена.\n\n" + ROLE_QUESTION
        if command == "/start":
            if profile.stage == "complete":
                return COMPLETED
            question = ROLE_QUESTION if profile.stage == "target_role" else ABOUT_QUESTION
            return HELP + "\n\n" + question
        if command == "/continue":
            if not profile.target_role:
                return "Сначала укажи желаемую должность.\n\n" + ROLE_QUESTION
            profile.stage = "complete"
            self.store.save(profile)
            return COMPLETED
        if text.startswith("/"):
            return "Неизвестная команда. Список команд: /help."

        if profile.stage == "target_role":
            profile.target_role = text
            profile.stage = "about"
            reply = "Должность сохранена.\n\n" + ABOUT_QUESTION
        elif profile.stage == "about":
            profile.about = text
            profile.stage = "complete"
            reply = COMPLETED
        else:
            return "Анкета уже сохранена. Посмотреть: /profile. Заполнить заново: /restart."

        self.store.save(profile)
        return reply

