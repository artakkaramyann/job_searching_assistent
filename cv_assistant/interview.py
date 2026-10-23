"""Команды и соединение интерфейсов с LangGraph; здесь нет вызовов Telegram."""

from langgraph.checkpoint.sqlite import SqliteSaver

from cv_assistant.graph import ABOUT_QUESTION, COMPLETED, ROLE_QUESTION, build_graph, initial_state
from cv_assistant.llm import AnalysisError, Analyzer
from cv_assistant.models import InterviewState, LABELS
from cv_assistant.memory import clean_facts, clean_summaries, find_gaps, required_topics
from cv_assistant.languages import extract_languages
from cv_assistant.reviewer import Reviewer, review_text
from cv_assistant.storage import Profile, ProfileStore

MENU = "📋 Меню\n\nПродолжим анкету или посмотрим, что уже готово? Выбери кнопку внизу."
HELP = "🚀 Помогу собрать сведения для резюме.\n\n💬 Продолжить — начать, дополнить или повторить обработку.\n👤 Профиль — твоя анкета.\n📝 Рекомендации — что улучшить.\n\nОтветы сохраняются и передаются ИИ для анализа."
CONTINUE_PHRASES = {"продолжить", "давай продолжим", "надо продолжать", "хватит вопросов", "завершить опрос"}
RETRY_MESSAGE = "⏳ Ответы сохранены. Не удалось закончить обработку — нажми «💬 Продолжить» чуть позже."


def current_view(state: InterviewState) -> InterviewState:
    facts = clean_facts(state["facts"], state["turns"])
    gaps = find_gaps(state["target_role"], facts, state.get("gaps", []), state["turns"])
    # Не показываем старое ревью, если факты изменились при проверке старой анкеты.
    review = state.get("review") if facts == state["facts"] else None
    return {**state, "facts": facts, "gaps": gaps, "review": review,
            "summaries": clean_summaries(state.get("summaries", []), facts, state["turns"]),
            "languages": extract_languages(state["turns"], state.get("languages", []))}


def profile_text(state: InterviewState) -> str:
    state = current_view(state)
    lines = [f"👤 Твой профиль\n🎯 {state['target_role'] or 'Должность пока не указана'}"]
    for topic, label in LABELS.items():
        if topic == "languages" and state.get("languages"):
            lines.append("🌍 Языки:\n" + "\n".join(
                f"• {item['name']} — {item['level'] or 'уровень не указан'}" for item in state["languages"]))
            continue
        facts = [fact["text"] for fact in state["facts"] if fact["topic"] == topic]
        summary = next((s["text"] for s in state.get("summaries", []) if s["topic"] == topic), None)
        if summary:
            lines.append(f"{label}: {summary}")
        elif facts:
            lines.append(f"{label}: " + "; ".join(facts))
    known = {fact["topic"] for fact in state["facts"]}
    if state.get("languages"):
        known.add("languages")
    missing = [LABELS[topic] for topic in required_topics(state["target_role"]) if topic not in known]
    if missing:
        lines.append("🔎 Ещё уточним: " + ", ".join(missing))
    if state["needs_analysis"]:
        lines.append(RETRY_MESSAGE)
    return "\n\n".join(lines)


def history_text(state: InterviewState) -> str:
    lines = ["Исходные ответы:"]
    lines.extend(f"{turn['question']}\n{turn['answer']}" for turn in state["turns"])
    if not state["turns"]:
        lines.append("Пока нет ответов.")
    return "\n\n".join(lines)


class Interview:
    def __init__(self, store: ProfileStore, analyzer: Analyzer, max_questions: int = 5,
                 reviewer: Reviewer | None = None):
        self.store = store
        self.analyzer = analyzer
        self.max_questions = max_questions
        self.reviewer = reviewer
        self.sync_profiles()

    def sync_profiles(self) -> None:
        """Заполняем читаемую таблицу из актуальных checkpoints, без вызовов ИИ."""
        for user_id in self.store.checkpoint_users():
            self.get_state(user_id)

    def _legacy_state(self, user_id: int) -> InterviewState:
        old = self.store.get(user_id)
        return initial_state(old.target_role, old.about, old.stage)

    def get_state(self, user_id: int) -> InterviewState:
        with SqliteSaver.from_conn_string(str(self.store.path)) as saver:
            graph = build_graph(self.analyzer, saver, self.max_questions, self.reviewer)
            state = graph.get_state({"configurable": {"thread_id": str(user_id)}}).values
            if state:
                result = current_view({**initial_state(), **state})
                self.store.save_state(user_id, result)
                return result
            return self._legacy_state(user_id)

    def handle(self, user_id: int, text: str) -> str:
        text = text.strip()
        if not text:
            return "Пришли, пожалуйста, текстовый ответ."
        if len(text) > 4000:
            return "Сократи ответ до 4000 символов и отправь снова."
        command = text.split()[0].split("@")[0].lower()
        if command == "/help":
            return HELP
        if command == "/menu":
            return MENU
        allowed = {"/start", "/restart", "/profile", "/continue", "/retry", "/resume", "/history", "/review", "/review_full", "/go"}
        if text.startswith("/") and command not in allowed:
            return "Открой 📋 Меню и выбери действие."
        finish = command == "/continue" or text.casefold().rstrip(".! ") in CONTINUE_PHRASES
        config = {"configurable": {"thread_id": str(user_id)}}
        with SqliteSaver.from_conn_string(str(self.store.path)) as saver:
            graph = build_graph(self.analyzer, saver, self.max_questions, self.reviewer)
            snapshot = graph.get_state(config)
            state = current_view({**initial_state(), **snapshot.values}) if snapshot.values else self._legacy_state(user_id)
            if snapshot.values:
                self.store.save_state(user_id, state)
            if command == "/go":
                command = "/retry" if snapshot.next else ("/resume" if state["stage"] == "complete" else "/start")
            if command == "/restart":
                # Очищаем старую анкету, чтобы она не восстановилась при следующем запуске.
                self.store.save(Profile(user_id=user_id))
                saver.delete_thread(str(user_id))
                return "🔄 Начинаем заново!\n\n" + ROLE_QUESTION
            if command == "/profile":
                return profile_text(state)
            if command == "/history":
                return history_text(state)
            if command in {"/review", "/review_full"}:
                if snapshot.next:
                    return RETRY_MESSAGE
                if state["stage"] != "complete":
                    return "Сначала нажми «✅ Завершить опрос», чтобы получить рекомендации."
                # Старое ревью с индексами цитат обновляем один раз по запросу пользователя.
                if state.get("review") is not None and "strengths" in state["review"]:
                    return review_text(state, full=command == "/review_full")
                finish = True
            if command == "/resume" and not state["target_role"]:
                return ROLE_QUESTION
            if finish and not state["target_role"]:
                return "Сначала укажи желаемую должность.\n\n" + ROLE_QUESTION
            if command == "/start":
                if snapshot.next:
                    return RETRY_MESSAGE
                return ("🚀 Привет! Помогу собрать резюме.\n\n" if not state["target_role"] else "") + (COMPLETED if state["stage"] == "complete" else state["question"])
            if state["stage"] == "complete" and command not in {"/resume", "/retry", "/review", "/review_full"}:
                return COMPLETED
            if snapshot.next and command not in {"/retry", "/resume"} and not finish:
                return RETRY_MESSAGE
            if command == "/retry" and not snapshot.next:
                if state["stage"] == "complete":
                    return COMPLETED
                return state["question"]
            if not finish and command not in {"/retry", "/resume"}:
                total = sum(len(turn["answer"]) for turn in state["turns"]) + len(text)
                if total > 12000:
                    return "Анкета стала слишком длинной. Сократи ответ или заверши опрос: /continue."
            try:
                if command == "/retry":
                    # None продолжает незавершённый узел; ответ повторно не добавляется.
                    result = graph.invoke(None, config)
                else:
                    data = {"incoming_text": text, "finish_requested": finish,
                            "action": "resume" if command == "/resume" else "answer"}
                    if not snapshot.values:
                        data = {**state, **data}
                    result = graph.invoke(data, config)
            except AnalysisError:
                return RETRY_MESSAGE
            finally:
                saved = graph.get_state(config).values
                if saved:
                    self.store.save_state(user_id, current_view({**initial_state(), **saved}))
            if command in {"/review", "/review_full"}:
                return review_text(result, full=command == "/review_full")
            return result["reply"]
