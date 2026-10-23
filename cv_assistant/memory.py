"""Проверка источников и пробелов, общая для интервью и ревью.

Текст анкеты берём из слов человека. Перефразированную моделью цитату
не считаем доказательством; отбрасываем только спорный факт, сохраняя диалог.
"""

import re
import unicodedata
import hashlib
import json

from cv_assistant.models import Gap, LABELS, REQUIRED_TOPICS
from cv_assistant.languages import extract_languages

QUESTIONS = {
    "experience": "Есть ли опыт работы или стажировок? Если есть, расскажи о своей роли и задачах.",
    "skills": "Какие навыки и инструменты ты уже применял на практике?",
    "projects": "Какой пример своей работы ты хотел бы показать работодателю?",
    "education": "Где и чему ты учишься? Можно указать курсы или самостоятельное обучение.",
    "languages": "Какими языками ты владеешь и на каком уровне?",
}
UNKNOWN = re.compile(
    r"не (?:были? |была |были )?(?:указан|упомянут|предоставлен|сообщен|сообщён)|"
    r"нет (?:информации|сведений|данных)|неизвестн|not (?:mentioned|specified|provided)", re.I
)
ABSENT = re.compile(r"\bнет\b|отсутств|не имею|не работал|без опыта|не хочу|не буду|no experience", re.I)
ALREADY_ANSWERED = re.compile(
    r"(?:уже|тоже|ранее)\s+(?:на\s+это\s+)?(?:отвечал|ответил|рассказывал|писал)|"
    r"(?:отвечал|ответил|рассказывал|писал)\s+(?:уже|ранее)|^отвечал[.! ]*$", re.I
)
TENTATIVE = re.compile(r"пытал|попытк|пробовал|искал|ищу|работал[аи]? над|планиру", re.I)
NO_EQUIPMENT = re.compile(
    r"(?:оборудован\w*|прибор\w*|инструмент\w*)\s+не\s+(?:использовал\w*|применял\w*)|"
    r"не\s+(?:использовал\w*|применял\w*)\s+(?:(?:никак\w*)\s+)?(?:оборудован\w*|прибор\w*|инструмент\w*)|"
    r"не\s+работал\w*\s+с\s+(?:оборудован\w*|прибор\w*|инструмент\w*)", re.I
)
TOPIC_MARKERS = {
    "experience": r"работал|работаю|работаем|опыт|стаж|intern|experience|employment|\d+\s*(?:лет|год)|повар|кондитер|готовлю",
    "projects": r"проект|project|созда|сдела|разработ|бот|приложен",
    "languages": r"язык|англий|русск|армян|немец|француз|испан|китай|english|russian|armenian|german|french|spanish|chinese",
}


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def answers_revision(turns: list[dict]) -> str:
    return hashlib.sha256(json.dumps(turns, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def repeat_only(text: str) -> bool:
    """Жалоба без новых сведений. Смешанный ответ с исправлением надо анализировать."""
    if not ALREADY_ANSWERED.search(text):
        return False
    rest = ALREADY_ANSWERED.sub(" ", text.casefold())
    filler = {"я", "это", "тоже", "уже", "же", "об", "этом", "на", "вам", "тебе",
              "вопрос", "мой", "единственный", "проект"}
    return all(word in filler for word in re.findall(r"\w+", rest))


def merge_summaries(previous: list[dict], proposed: list[dict], old_facts: list[dict], facts: list[dict],
                   turns: list[dict] = ()) -> list[dict]:
    """Новая версия раздела заменяет старую; неизменённые разделы не теряются.

    При исправлении фактов прежний summary нельзя оставлять как актуальный.
    """
    def sources(items, topic):
        return {normalize(f["evidence"]).casefold() for f in items if f["topic"] == topic}
    def unchanged(topic):
        old, new = sources(old_facts, topic), sources(facts, topic)
        if old == new:
            return True
        # Модель может разбить «Знаю Python и SQL» на две цитаты, не изменив сведения.
        # Отрицания и квалификаторы («изучаю», «пытался») остаются значимыми словами.
        def words(quotes):
            return set(re.findall(r"\w+", " ".join(quotes))) - {"я", "и", "также", "знаю", "умею"}
        fragments = (all(any(q in source for source in old) for q in new)
                     or all(any(q in source for source in new) for q in old))
        return bool(old and new) and fragments and words(old) == words(new)
    kept = [s for s in previous if unchanged(s["topic"])]
    result = {s["topic"]: s for s in clean_summaries(kept, facts, turns)}
    for summary in clean_summaries(proposed, facts, turns):
        result.setdefault(summary["topic"], summary)
    return list(result.values())


def clean_name(text: str) -> str:
    text = re.sub(r"^(?:меня зовут|мо[её] имя|my name is|i am)\s+", "", text, flags=re.I)
    # Не захватываем продолжение «Артак я студент…» в поле имени.
    match = re.match(r"[^\W\d_][\w'’-]*(?:\s+[A-ZА-ЯЁ][\w'’-]*){0,2}", text)
    return match.group() if match else text


def clean_facts(facts: list[dict], turns: list[dict]) -> list[dict]:
    cleaned = []
    seen = set()
    for fact in facts:
        topic = fact.get("topic")
        text = normalize(fact.get("text", ""))
        evidence = normalize(fact.get("evidence", ""))
        if topic not in LABELS or not evidence or UNKNOWN.search(text) or UNKNOWN.search(evidence):
            continue
        if repeat_only(evidence):
            continue
        sources = [turn for turn in turns if evidence.casefold() in normalize(turn["answer"]).casefold()]
        if not sources:
            continue
        # «Я студент» не доказывает «опыта нет». Нет упоминания != явное отсутствие.
        if ABSENT.search(text) and not ABSENT.search(evidence):
            continue
        if topic in TOPIC_MARKERS and not re.search(TOPIC_MARKERS[topic], evidence, re.I):
            if not any(turn["topic"] == topic for turn in sources):
                continue
        # Сохраняем исходное написание. Так YSU не превратится в выдуманный вуз.
        source = normalize(sources[-1]["answer"])
        match = re.search(re.escape(evidence), source, re.I)
        if match is None:
            continue
        quote = match.group()
        display = quote
        if text and text.casefold() in quote.casefold():
            match = re.search(re.escape(text), quote, re.I)
            if match is not None:
                display = match.group()
        if topic == "name":
            display = clean_name(display)
        if TENTATIVE.search(quote) and not TENTATIVE.search(display):
            display = quote
        identity = (topic, display.casefold())
        if identity not in seen:
            cleaned.append({"topic": topic, "text": display, "evidence": quote})
            seen.add(identity)
    return cleaned


def clean_summaries(summaries: list[dict], facts: list[dict], turns: list[dict] = ()) -> list[dict]:
    """Храним текст агента отдельно от цитат; неподтверждённые ссылки отбрасываем.

    Проверка источников не заменяет семантическую проверку человеком.
    """
    result = {}
    answers = [normalize(turn["answer"]).casefold() for turn in turns]
    for summary in summaries:
        topic = summary["topic"]
        sources = [normalize(f["evidence"]).casefold() for f in facts if f["topic"] == topic]
        quotes = summary["evidence"]
        def supported(quote):
            q = normalize(quote).casefold()
            return bool(q) and (any(q in s for s in sources) or
                                (any(q in answer for answer in answers) and any(s in q for s in sources)))
        if not quotes or not all(supported(q) for q in quotes):
            continue
        cited_sources = " ".join(quotes) + " " + " ".join(
            s for s in sources if any(normalize(q).casefold() in s for q in quotes))
        if not summary_preserves_claims(summary["text"], cited_sources):
            continue
        result[topic] = summary
    return list(result.values())


def summary_preserves_claims(text: str, sources: str) -> bool:
    """Общие ограниченные проверки summary; не доказательство его смысла."""
    if ALREADY_ANSWERED.search(text):
        return False
    if TENTATIVE.search(sources) and not TENTATIVE.search(text):
        return False
    if re.search(r"учусь|учится|студент", sources, re.I) and re.search(
            r"окончил|закончил|выпускник|получил\w* диплом", text, re.I):
        return False
    return set(re.findall(r"\d+", text)) <= set(re.findall(r"\d+", sources))


def question_intent(question: str) -> str | None:
    # Оборудование может быть ошибочно помечено моделью и как skills, и как experience.
    if re.search(r"инструмент|оборудован|прибор|спектрометр|лазер[а-я]*\b", question, re.I):
        return "equipment"
    return None


def already_answered(gap: dict, turns: list[dict]) -> bool:
    intent = question_intent(gap["question"])
    if (intent == "equipment" or gap["key"] == "professional_tools") and any(
            NO_EQUIPMENT.search(turn["answer"]) for turn in turns):
        return True
    for turn in turns:
        if turn["topic"] == "about":
            continue
        if turn["topic"] == gap["topic"]:
            if turn.get("key", "overview") == gap["key"] or repeat_only(turn["answer"]):
                return True
        if intent and question_intent(turn["question"]) == intent:
            return True
    return False


def gap_id(gap: dict) -> str:
    return f"{gap['topic']}:{gap['key']}"


def is_ml_role(role: str) -> bool:
    return bool(re.search(r"machine learning|\bml\b|data scien|машинн|deep learning|ai engineer", role, re.I))


def is_technical_role(role: str) -> bool:
    return is_ml_role(role) or bool(re.search(
        r"программист|разработчик|developer|software|devops|backend|frontend|data engineer", role, re.I))


def required_topics(role: str) -> tuple:
    portfolio = is_technical_role(role) or bool(re.search(r"дизайн|фотограф|архитектор|designer", role, re.I))
    return REQUIRED_TOPICS if portfolio else tuple(t for t in REQUIRED_TOPICS if t != "projects")


def relevant_question(role: str, gap: dict) -> bool:
    if not is_technical_role(role):
        if gap["key"] in {"programming_languages", "libraries"}:
            return False
        if re.search(r"программиров|библиотек|pytorch|tensorflow|python|java|software", gap["question"], re.I):
            return False
    if gap["topic"] == "projects" and "projects" not in required_topics(role):
        # Примеры работ допустимы, обязательные «учебные проекты» для любой профессии — нет.
        if re.search(r"учебн.*проект", gap["question"], re.I):
            return False
    return True


def find_gaps(role: str, facts: list[dict], proposed: list[dict], turns: list[dict]) -> list[dict]:
    """Предпочитаем вопросы ИИ. Шаблон — запасной вариант для пропущенной темы."""
    known = {fact["topic"] for fact in facts}
    languages = extract_languages(turns)
    if languages:
        known.add("languages")
    gaps = []
    unresolved = [item for item in languages if not item["level"]]
    if unresolved:
        suggested = next((item.get("suggested_level") for item in unresolved if item.get("suggested_level")), None)
        names = ", ".join(item["name"] for item in unresolved)
        question = (f"Правильно понимаю, что уровень «{suggested}» относится ко всем: {names}?"
                    if suggested else f"Уточни уровень для каждого языка: {names}.")
        clarification = Gap(topic="languages", key="language_levels", question=question).model_dump()
        if not already_answered(clarification, turns):
            gaps.append(clarification)
    valid_pairs = {("skills", "programming_languages"), ("skills", "libraries"),
                   ("skills", "professional_tools"), ("projects", "project_details"),
                   ("experience", "experience_details"), ("education", "education_details"),
                   ("languages", "language_levels")}
    for gap in proposed:
        if already_answered(gap, turns):
            continue
        if not relevant_question(role, gap):
            continue
        if gap["key"] == "language_levels" and languages and all(x["level"] for x in languages):
            continue
        if gap["key"] == "overview":
            if gap["topic"] in known:
                continue
        elif (gap["topic"], gap["key"]) not in valid_pairs:
            continue
        topic_facts = [f for f in facts if f["topic"] == gap["topic"]]
        if topic_facts and all(ABSENT.search(f["evidence"]) for f in topic_facts):
            continue
        declined = any(t["topic"] == gap["topic"] and t.get("key") == gap["key"]
                       and re.search(r"не помню|не знаю|не хочу|пропуст", t["answer"], re.I) for t in turns)
        if not declined and not any(gap_id(g) == gap_id(gap) for g in gaps):
            gaps.append(gap)

    for topic in required_topics(role):
        if topic not in known and not any(g["topic"] == topic for g in gaps):
            question = QUESTIONS[topic]
            if topic == "projects" and is_ml_role(role):
                question = "Есть ли у тебя ML-проект? Какую задачу ты решал и что сделал сам?"
            fallback = Gap(topic=topic, key="overview", question=question).model_dump()
            if not already_answered(fallback, turns):
                gaps.append(fallback)

    # Резервные уточнения только для ML. Предложение модели имеет приоритет.
    skill_text = " ".join(f["text"] for f in facts if f["topic"] == "skills")
    if is_ml_role(role):
        for key, pattern, question in [
            ("programming_languages", r"\b(?:python|java|julia|r|sql|scala)\b",
             "Какие языки программирования ты применял в ML?"),
            ("libraries", r"pytorch|tensorflow|scikit|sklearn|pandas|numpy|keras|xgboost|catboost|jax|langgraph|langchain",
             "Какие библиотеки для ML и обработки данных ты применял на практике?"),
        ]:
            answered = any(t.get("key") == key and t["topic"] == "skills" for t in turns)
            if not re.search(pattern, skill_text, re.I) and not answered and not any(g["key"] == key for g in gaps):
                gaps.append(Gap(topic="skills", key=key, question=question).model_dump())
    return gaps
