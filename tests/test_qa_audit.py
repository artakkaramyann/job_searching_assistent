"""Регрессии, найденные независимым QA-аудитом; без сети и рабочей базы."""

import pytest

from cv_assistant.languages import extract_languages


def turn(answer, topic="languages", key="overview"):
    return {"topic": topic, "key": key, "question": "Какими языками владеешь?", "answer": answer}


@pytest.mark.parametrize("answer, forbidden_level", [
    ("Английский не родной", "родной"),
    ("На английском не говорю свободно", "свободно"),
    ("Английский не B2, пока не могу оценить уровень", "B2"),
])
def test_denied_language_level_is_not_claimed(answer, forbidden_level):
    languages = extract_languages([turn(answer)])
    assert not any(item["name"] == "Английский" and item["level"] == forbidden_level for item in languages)


def test_later_denial_removes_old_language_level():
    languages = extract_languages([
        turn("Английский B2, русский родной"),
        turn("Исправление: английского не знаю"),
    ])
    assert not any(item["name"] == "Английский" and item["level"] == "B2" for item in languages)
    assert any(item["name"] == "Русский" and item["level"] == "родной" for item in languages)


def test_qualified_confirmation_does_not_confirm_all_languages():
    languages = extract_languages([
        turn("Русский и английский знаю как носитель"),
        {**turn("Да, но не на всех", key="language_levels"),
         "question": "Этот уровень относится ко всем языкам?"},
    ])
    assert not all(item["level"] == "как носитель (самооценка)" for item in languages)


def test_prefix_level_belongs_to_first_language_in_multi_language_list():
    languages = extract_languages([turn("Родной армянский, русский B2, английский B1")])
    levels = {item["name"]: item["level"] for item in languages}
    assert levels == {"Армянский": "родной", "Русский": "B2", "Английский": "B1"}


def test_denied_native_level_keeps_explicit_b1():
    levels = extract_languages([turn("Английский не родной, уровень B1")])
    assert levels[0]["level"] == "B1"


def test_prefix_for_next_language_does_not_leak_into_previous_language():
    levels = {x["name"]: x["level"] for x in extract_languages([turn("Русский, родной армянский, английский B1")])}
    assert levels == {"Русский": None, "Армянский": "родной", "Английский": "B1"}


def test_mixed_repeat_complaint_can_correct_facts(tmp_path):
    from cv_assistant.interview import Interview
    from cv_assistant.models import Analysis, Fact, Gap
    from cv_assistant.storage import ProfileStore
    correction = "Я уже писал о навыках, но исправляю: знаю только SQL"

    class Model:
        def analyze(self, role, turns):
            if len(turns) == 1:
                return Analysis(facts=[Fact(topic="skills", text="Python", evidence="Python")],
                                gaps=[Gap(topic="skills", key="professional_tools", question="Какие инструменты знаешь?")])
            return Analysis(facts=[Fact(topic="skills", text="знаю только SQL", evidence=correction)], gaps=[])

    interview = Interview(ProfileStore(tmp_path / "db.sqlite3"), Model())
    interview.handle(1, "Developer")
    interview.handle(1, "Python")
    interview.handle(1, correction)
    profile = interview.handle(1, "/profile")
    assert "SQL" in profile and "Python" not in profile
    assert interview.get_state(1)["facts"][0]["text"] == "знаю только SQL"


def test_summary_survives_fact_fragmentation_but_not_changed_qualification():
    from cv_assistant.memory import merge_summaries
    from cv_assistant.models import Fact
    summary = {"topic": "skills", "text": "Знает Python и SQL.", "evidence": ["Знаю Python и SQL"]}
    old = [Fact(topic="skills", text="Знаю Python и SQL", evidence="Знаю Python и SQL").model_dump()]
    new = [Fact(topic="skills", text=s, evidence=s).model_dump() for s in ["Python", "SQL"]]
    turns = [turn("Знаю Python и SQL", "about"), turn("Русский родной")]
    assert merge_summaries([summary], [], old, new, turns) == [summary]
    corrected = [Fact(topic="skills", text=s, evidence=s).model_dump() for s in ["Изучаю Python", "SQL"]]
    assert merge_summaries([summary], [], old, corrected, turns + [turn("Изучаю Python")]) == []
