import json
import sqlite3

from cv_assistant.interview import Interview
from cv_assistant.llm import AnalysisError, OfflineInterviewer
from cv_assistant.memory import clean_summaries, merge_summaries
from cv_assistant.models import Analysis, Fact, Gap, SectionSummary
from cv_assistant.storage import ProfileStore


EDUCATION = "Я учусь в YSU 3 курс"
PROJECT = "Делал учебный проект на MNIST"


def analyzed():
    return Analysis(
        facts=[Fact(topic="education", text=EDUCATION, evidence=EDUCATION),
               Fact(topic="projects", text=PROJECT, evidence=PROJECT)],
        gaps=[Gap(topic="education", key="education_details", question="Какое направление изучаешь в YSU?")],
        summaries=[SectionSummary(topic="education", text="Учится на 3-м курсе YSU.", evidence=[EDUCATION]),
                   SectionSummary(topic="projects", text="Выполнял учебный проект на MNIST.", evidence=[PROJECT])],
    )


class Model:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = 0

    def analyze(self, target_role, turns):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def row(path, user_id):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return dict(db.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone())


def test_profile_stable_after_resume_finish_and_process_restart(tmp_path):
    store = ProfileStore(tmp_path / "db.sqlite3")
    model = Model(analyzed())  # Второй вызов модели был бы ошибкой: новых ответов нет.
    interview = Interview(store, model)
    interview.handle(1, "ML инженер")
    interview.handle(1, EDUCATION + ". " + PROJECT)
    interview.handle(1, "/continue")
    before = interview.handle(1, "/profile")
    for _ in range(2):
        assert "направление" in interview.handle(1, "/go")
        interview.handle(1, "/continue")
        assert interview.handle(1, "/profile") == before
    restarted = Interview(ProfileStore(store.path), model)
    assert restarted.handle(1, "/profile") == before
    assert model.calls == 1
    saved = row(store.path, 1)
    assert saved["stage"] == "complete"
    assert saved["target_role"] == "ML инженер"
    assert len(json.loads(saved["turns_json"])) == 1
    assert json.loads(saved["summaries_json"]) == restarted.get_state(1)["summaries"]
    assert json.loads(saved["review_json"]) is not None
    assert saved["updated_at"]


def test_new_answer_runs_analysis_and_keeps_unchanged_sections(tmp_path):
    initial = analyzed()
    changed = Analysis(facts=[*initial.facts, Fact(topic="skills", text="Python", evidence="Python")], gaps=[], summaries=[])
    model = Model(initial, changed)
    interview = Interview(ProfileStore(tmp_path / "db.sqlite3"), model)
    interview.handle(1, "ML инженер")
    interview.handle(1, EDUCATION + ". " + PROJECT)
    interview.handle(1, "Python")
    assert model.calls == 2
    profile = interview.handle(1, "/profile")
    assert "Образование: Учится на 3-м курсе YSU." in profile
    assert "Проекты: Выполнял учебный проект на MNIST." in profile
    assert "Python" in profile


def test_changed_fact_does_not_keep_outdated_summary():
    old = analyzed()
    facts = [Fact(topic="education", text="Перевёлся в другой вуз", evidence="Перевёлся в другой вуз").model_dump()]
    assert merge_summaries([s.model_dump() for s in old.summaries], [],
                          [f.model_dump() for f in old.facts], facts) == []


def test_longer_summary_quote_can_cite_split_facts_only_if_present_in_answer():
    facts = [Fact(topic="education", text="YSU", evidence="YSU").model_dump(),
             Fact(topic="education", text="3 курс", evidence="3 курс").model_dump()]
    summary = {"topic": "education", "text": "Учится на 3-м курсе YSU.", "evidence": [EDUCATION]}
    assert clean_summaries([summary], facts, [{"answer": EDUCATION}]) == [summary]
    assert clean_summaries([summary], facts, [{"answer": "YSU"}]) == []


def test_sql_projection_saves_answer_even_when_model_fails(tmp_path):
    store = ProfileStore(tmp_path / "db.sqlite3")
    interview = Interview(store, Model(AnalysisError("Лимит Groq")))
    interview.handle(1, "Повар")
    assert row(store.path, 1)["target_role"] == "Повар"
    interview.handle(1, "Готовлю торты")
    assert json.loads(row(store.path, 1)["turns_json"])[0]["answer"] == "Готовлю торты"
    assert Interview(ProfileStore(store.path), OfflineInterviewer()).get_state(1)["needs_analysis"]


def test_schema_migration_backfill_and_restart_clear_only_selected_user(tmp_path):
    path = tmp_path / "db.sqlite3"
    # Существующая таблица первой версии без JSON-колонок.
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE profiles (user_id INTEGER PRIMARY KEY, stage TEXT NOT NULL, target_role TEXT NOT NULL, about TEXT NOT NULL)")
        db.execute("INSERT INTO profiles VALUES (9, 'about', 'Повар', 'Старые ответы')")
    store = ProfileStore(path)
    assert store.get(9).about == "Старые ответы"
    interview = Interview(store, Model(analyzed()))
    interview.handle(1, "Физик")
    interview.handle(1, EDUCATION + ". " + PROJECT)
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM profiles WHERE user_id=1")  # Проекция отстала; checkpoints сохранены.
    restarted = Interview(ProfileStore(path), OfflineInterviewer())
    assert json.loads(row(path, 1)["summaries_json"])
    restarted.handle(1, "/restart")
    saved = row(path, 1)
    for field in ["turns_json", "facts_json", "summaries_json", "languages_json"]:
        assert json.loads(saved[field]) == []
    assert saved["target_role"] == ""
    assert store.get(9).about == "Старые ответы"
    assert Interview(ProfileStore(path), OfflineInterviewer()).get_state(1)["turns"] == []
