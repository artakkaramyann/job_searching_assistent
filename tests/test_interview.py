import pytest

from cv_assistant.graph import ABOUT_QUESTION, COMPLETED, ROLE_QUESTION
from cv_assistant.interview import Interview
from cv_assistant.llm import AnalysisError, OfflineInterviewer
from cv_assistant.models import Analysis, Fact, Gap, REQUIRED_TOPICS
from cv_assistant.storage import Profile, ProfileStore


class ScriptedAnalyzer:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def analyze(self, target_role, turns):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def result(topics=(), question_topic=None, question="", evidence="Учусь Python"):
    return Analysis(
        facts=[Fact(topic=t, text=evidence, evidence=evidence) for t in topics],
        gaps=([Gap(topic=question_topic, key="overview", question=question)] if question_topic else []),
    )


def make_interview(tmp_path, analyzer=None, max_questions=5):
    return Interview(ProfileStore(tmp_path / "profiles.sqlite3"), analyzer or OfflineInterviewer(), max_questions)


def test_full_offline_interview(tmp_path):
    interview = make_interview(tmp_path)
    assert ROLE_QUESTION in interview.handle(1, "/start")
    assert ABOUT_QUESTION in interview.handle(1, "Junior Python Developer")
    interview.handle(1, "Учусь Python")
    for answer in ["Опыта нет", "Python и SQL", "Telegram-бот", "Курсы", "Русский родной, английский B1"]:
        reply = interview.handle(1, answer)
    assert COMPLETED in reply
    state = interview.get_state(1)
    assert state["stage"] == "complete"
    assert state["completion_reason"] == "enough"
    assert state["about"] == "Учусь Python"
    assert len(state["turns"]) == 6


def test_full_narrative_ends_without_redundant_questions(tmp_path):
    values = {"experience": "Работал разработчиком", "skills": "Python",
              "projects": "Создал Telegram-бота", "education": "Курсы", "languages": "Русский родной"}
    analysis = Analysis(facts=[Fact(topic=topic, text=text, evidence=text) for topic, text in values.items()], gaps=[])
    analyzer = ScriptedAnalyzer(analysis)
    interview = make_interview(tmp_path, analyzer)
    interview.handle(1, "Developer")
    assert COMPLETED in interview.handle(1, ". ".join(values.values()))
    assert interview.get_state(1)["asked_topics"] == []
    assert analyzer.calls == 1


def test_question_only_for_missing_topic(tmp_path):
    # Если модель предлагает повторить уже известное, граф выбирает пробел.
    analyzer = ScriptedAnalyzer(result(["skills"], "skills", "Повтори навыки"))
    interview = make_interview(tmp_path, analyzer)
    interview.handle(1, "Developer")
    reply = interview.handle(1, "Учусь Python")
    assert "Повтори навыки" not in reply
    assert interview.get_state(1)["question_topic"] == "experience"


def test_restart_program_keeps_question_and_profile(tmp_path):
    interview = make_interview(tmp_path)
    interview.handle(1, "Developer")
    interview.handle(1, "Учусь Python")
    restarted = make_interview(tmp_path)
    assert restarted.get_state(1)["question"] in restarted.handle(1, "/start")
    restarted.handle(1, "Опыта нет")
    assert "Учусь Python" in restarted.handle(1, "/history")
    assert "Опыта нет" in restarted.handle(1, "/profile")


def test_users_and_restart_are_isolated(tmp_path):
    interview = make_interview(tmp_path)
    interview.handle(1, "Developer")
    interview.handle(2, "Designer")
    interview.handle(1, "/restart")
    assert interview.get_state(1)["target_role"] == ""
    assert interview.get_state(2)["target_role"] == "Designer"
    assert interview.get_state(3)["target_role"] == ""


@pytest.mark.parametrize("text", ["/continue", "Продолжить!", "Хватит вопросов"])
def test_continue_requires_role_and_does_not_call_model(tmp_path, text):
    analyzer = ScriptedAnalyzer()
    interview = make_interview(tmp_path, analyzer)
    assert ROLE_QUESTION in interview.handle(1, text)
    interview.handle(1, "Developer")
    assert COMPLETED in interview.handle(1, text)
    assert interview.get_state(1)["about"] == ""
    assert analyzer.calls == 0


def test_commands_do_not_replace_answers_or_use_model(tmp_path):
    analyzer = ScriptedAnalyzer()
    interview = make_interview(tmp_path, analyzer)
    interview.handle(1, "Developer")
    for command in ["/start@my_bot", "/help", "/unknown", "/profile", "/retry", "   "]:
        interview.handle(1, command)
    state = interview.get_state(1)
    assert state["stage"] == "about"
    assert state["target_role"] == "Developer"
    assert state["turns"] == []
    assert analyzer.calls == 0


def test_limit_stops_loop_honestly(tmp_path):
    analyzer = ScriptedAnalyzer(result(), result())
    interview = make_interview(tmp_path, analyzer, max_questions=1)
    interview.handle(1, "Developer")
    interview.handle(1, "Учусь Python")
    reply = interview.handle(1, "Не знаю")
    assert "ещё есть пробелы" in reply.lower()
    assert interview.get_state(1)["completion_reason"] == "limit"
    interview.handle(1, "Случайное сообщение")
    assert analyzer.calls == 2


def test_failed_analysis_resumes_after_restart_without_duplicating_answer(tmp_path):
    first = make_interview(tmp_path, ScriptedAnalyzer(AnalysisError("Лимит Groq")))
    first.handle(1, "Developer")
    assert "Продолжить" in first.handle(1, "Учусь Python")
    assert first.get_state(1)["needs_analysis"] is True
    second = make_interview(tmp_path, ScriptedAnalyzer(result(["skills"])))
    assert "Продолжить" in second.handle(1, "/start")
    assert "Продолжить" in second.handle(1, "Ещё одно сообщение")
    second.handle(1, "/retry")
    state = second.get_state(1)
    assert len(state["turns"]) == 1
    assert state["turns"][0]["answer"] == "Учусь Python"
    assert state["stage"] == "followup"
    assert state["needs_analysis"] is False


def test_can_finish_after_failure_without_retrying_api(tmp_path):
    analyzer = ScriptedAnalyzer(AnalysisError("Лимит Groq"))
    interview = make_interview(tmp_path, analyzer)
    interview.handle(1, "Developer")
    interview.handle(1, "Учусь Python")
    assert COMPLETED in interview.handle(1, "/continue")
    assert interview.get_state(1)["stage"] == "complete"
    assert interview.get_state(1)["turns"][0]["answer"] == "Учусь Python"
    assert analyzer.calls == 1


def test_rejects_evidence_not_in_user_answers(tmp_path):
    analyzer = ScriptedAnalyzer(result(["experience"], evidence="Работал 10 лет"))
    interview = make_interview(tmp_path, analyzer)
    interview.handle(1, "Developer")
    reply = interview.handle(1, "Учусь Python")
    assert "/retry" not in reply
    assert interview.get_state(1)["stage"] == "followup"
    assert interview.get_state(1)["facts"] == []


@pytest.mark.parametrize("stage", ["about", "complete"])
def test_old_profiles_are_preserved(tmp_path, stage):
    store = ProfileStore(tmp_path / "profiles.sqlite3")
    store.save(Profile(user_id=1, stage=stage, target_role="Developer", about="Старый рассказ"))
    interview = Interview(store, OfflineInterviewer())
    assert "Старый рассказ" in interview.handle(1, "/history")
    assert interview.get_state(1)["stage"] == stage
    if stage == "about":
        interview.handle(1, "Новый рассказ")
        assert interview.get_state(1)["target_role"] == "Developer"
    interview.handle(1, "/restart")
    restarted = Interview(store, OfflineInterviewer())
    assert restarted.get_state(1)["turns"] == []
    assert restarted.get_state(1)["target_role"] == ""
