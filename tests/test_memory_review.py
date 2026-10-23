from langgraph.checkpoint.sqlite import SqliteSaver

from cv_assistant.graph import build_graph, initial_state
from cv_assistant.interview import Interview, profile_text
from cv_assistant.llm import AnalysisError, OfflineInterviewer
from cv_assistant.memory import clean_facts
from cv_assistant.models import Analysis, Fact, ProfileReview
from cv_assistant.storage import ProfileStore

STORY = "Меня зовут Артак я студент 3 курса YSU факультет физика, изучаю ML, Deep Learning, AI Agentic system ищу работу в этих сферах"


def fact(topic, text, evidence=None):
    return Fact(topic=topic, text=text, evidence=evidence or text)


def screenshot_facts():
    return [
        fact("name", "Меня зовут Артак"),
        fact("education", "Ярославский государственный университет", "студент 3 курса YSU факультет физика"),
        fact("skills", "Изучает машинное обучение", "изучаю ML, Deep Learning, AI Agentic system"),
        fact("experience", "Опыт работы отсутствует", "я студент 3 курса YSU"),
        fact("projects", "Учебные проекты не упомянуты", "изучаю ML"),
        fact("languages", "Языки не указаны", "изучаю ML"),
        fact("skills", "PyTorch", "Применял PyTorch"),
    ]


class Responses:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def analyze(self, target_role, turns):
        self.calls += 1
        return next(self.responses)


class Reviews:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def review(self, target_role, facts, gaps):
        self.calls += 1
        self.facts, self.gaps = facts, gaps
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def test_screenshot_regression_no_fake_completeness_or_blocked_dialogue(tmp_path):
    analyzer = Responses(Analysis(facts=screenshot_facts(), gaps=[]))
    interview = Interview(ProfileStore(tmp_path / "db.sqlite3"), analyzer)
    interview.handle(1, "Machine Learning инженер")
    reply = interview.handle(1, STORY)
    state = interview.get_state(1)
    assert state["stage"] == "followup"
    assert "/retry" not in reply
    gaps = {(gap["topic"], gap["key"]) for gap in state["gaps"]}
    assert {("projects", "overview"), ("languages", "overview"),
            ("experience", "overview"), ("skills", "programming_languages"),
            ("skills", "libraries")} <= gaps
    profile = interview.handle(1, "/profile")
    assert "Имя: Артак" in profile
    assert "Имя: Меня зовут" not in profile
    assert "Ярослав" not in profile
    assert "YSU" in profile
    assert "Опыт работы отсутствует" not in profile
    assert "PyTorch" not in profile
    assert "Исходные ответы" not in profile
    assert STORY not in profile
    assert STORY in interview.handle(1, "/history")


def test_normalized_quotes_accepted_without_accepting_paraphrases():
    turns = [{"topic": "about", "answer": "Изучаю   Python\nи SQL", "question": "О себе"}]
    facts = [fact("skills", "python и sql", "изучаю Python и SQL").model_dump(),
             fact("skills", "TensorFlow", "изучаю TensorFlow").model_dump()]
    cleaned = clean_facts(facts, turns)
    assert len(cleaned) == 1
    assert cleaned[0]["text"] == "Python и SQL"


def test_role_specific_skill_details_even_with_all_sections_present(tmp_path):
    values = {"experience": "Опыта нет", "skills": "Изучаю ML", "projects": "Создал учебный проект",
              "education": "Курсы", "languages": "Русский родной"}
    facts = [fact(topic, text) for topic, text in values.items()]
    analyzer = Responses(Analysis(facts=facts, gaps=[]),
                         Analysis(facts=[*facts, fact("skills", "Python")], gaps=[]))
    interview = Interview(ProfileStore(tmp_path / "db.sqlite3"), analyzer)
    interview.handle(1, "ML engineer")
    assert "языки программирования" in interview.handle(1, ". ".join(values.values()))
    reply = interview.handle(1, "Python")
    assert "библиотеки" in reply
    state = interview.get_state(1)
    assert state["stage"] == "followup"
    assert state["asked_keys"] == ["skills:programming_languages", "skills:libraries"]


def test_resume_old_finished_checkpoint_keeps_answers_and_reanalyzes(tmp_path):
    store = ProfileStore(tmp_path / "db.sqlite3")
    analyzer = Responses(Analysis(facts=screenshot_facts(), gaps=[]))
    with SqliteSaver.from_conn_string(str(store.path)) as saver:
        graph = build_graph(OfflineInterviewer(), saver)
        old_state = initial_state("ML engineer", STORY, "complete")
        old_state["facts"] = [f.model_dump() for f in screenshot_facts()]
        # Эмуляция схемы checkpoint предыдущей версии приложения.
        for key in ["review", "gaps", "asked_keys", "question_key", "action"]:
            old_state.pop(key)
        graph.update_state({"configurable": {"thread_id": "1"}}, old_state, as_node="review_profile")
    interview = Interview(store, analyzer)
    assert "Ярослав" not in interview.handle(1, "/profile")
    interview.handle(1, "/resume")
    assert interview.get_state(1)["stage"] == "followup"
    assert len(interview.get_state(1)["turns"]) == 1
    assert interview.get_state(1)["turns"][0]["answer"] == STORY
    assert analyzer.calls == 1


def test_reviewer_is_next_node_and_cached_and_uses_only_existing_facts(tmp_path):
    review = ProfileReview(strengths=[{"text": "Знает язык Python.", "source_fact_ids": [0]},
                                      {"text": "Необоснованное утверждение.", "source_fact_ids": [999, -1]}],
                           checks=["Уточнить название вуза"],
                           suggestions=["Описать вклад в проект"])
    reviewer = Reviews(review)
    analyzer = Responses(Analysis(facts=[fact("skills", "Python")], gaps=[]))
    interview = Interview(ProfileStore(tmp_path / "db.sqlite3"), analyzer, reviewer=reviewer)
    interview.handle(1, "Developer")
    interview.handle(1, "Python")
    reply = interview.handle(1, "/continue")
    assert "Анкета сохранена" in reply
    assert len(reply) < 200
    assert "Python" not in reply
    summary = interview.handle(1, "/review")
    assert "Python" in summary
    assert "Что уточнить" in summary
    assert interview.get_state(1)["review"]["strengths"] == [
        {"text": "Знает язык Python.", "source_fact_ids": [0]}]
    for _ in range(2):
        assert "Рекомендации для резюме" in interview.handle(1, "/review")
    assert reviewer.calls == 1
    assert analyzer.calls == 1


def test_review_failure_retries_only_review_even_after_restart(tmp_path):
    reviewer = Reviews(AnalysisError("Лимит Groq"))
    analyzer = Responses(Analysis(facts=[fact("skills", "Python")], gaps=[]))
    store = ProfileStore(tmp_path / "db.sqlite3")
    interview = Interview(store, analyzer, reviewer=reviewer)
    interview.handle(1, "Developer")
    interview.handle(1, "Python")
    assert "Продолжить" in interview.handle(1, "/continue")
    assert "Python" in interview.handle(1, "/profile")
    replacement = Reviews(ProfileReview(strengths=[], checks=[], suggestions=[]))
    restarted = Interview(store, analyzer, reviewer=replacement)
    assert "Анкета сохранена" in restarted.handle(1, "/go")
    assert "Рекомендации для резюме" in restarted.handle(1, "/review")
    assert replacement.calls == 1
    assert analyzer.calls == 1
    assert len(restarted.get_state(1)["turns"]) == 1


def test_resume_invalidates_review_without_clearing_history(tmp_path):
    analyzer = Responses(Analysis(facts=[fact("skills", "Python")], gaps=[]),
                         Analysis(facts=[fact("skills", "Python")], gaps=[]))
    interview = Interview(ProfileStore(tmp_path / "db.sqlite3"), analyzer)
    interview.handle(1, "Developer")
    interview.handle(1, "Python")
    interview.handle(1, "/continue")
    assert interview.get_state(1)["review"] is not None
    interview.handle(1, "/resume")
    assert interview.get_state(1)["review"] is None
    assert len(interview.get_state(1)["turns"]) == 1
