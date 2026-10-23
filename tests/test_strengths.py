import json
import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from cv_assistant.graph import build_graph
from cv_assistant.interview import Interview
from cv_assistant.llm import AnalysisError
from cv_assistant.models import Analysis, Fact, ProfileReview
from cv_assistant.reviewer import clean_review, review_text
from cv_assistant.storage import ProfileStore


FACTS = [
    Fact(topic="name", text="Артак", evidence="Меня зовут Артак"),
    Fact(topic="education", text="учусь YSU факультет физики 3 курс",
         evidence="учусь YSU факультет физики 3 курс"),
    Fact(topic="skills", text="изучаю Python и PyTorch", evidence="изучаю Python и PyTorch"),
]
STRENGTHS = [
    {"text": "Учится на третьем курсе физического факультета YSU.", "source_fact_ids": [1]},
    {"text": "Изучает Python и библиотеку PyTorch для работы в ML.", "source_fact_ids": [2]},
]


class Analyzer:
    def __init__(self):
        self.calls = 0

    def analyze(self, target_role, turns):
        self.calls += 1
        assert self.calls == 1, "Просмотр рекомендаций не должен повторять анализ интервью"
        return Analysis(facts=FACTS, gaps=[])


class Reviewer:
    def __init__(self):
        self.calls = 0
        self.fail = False

    def review(self, target_role, facts, gaps):
        self.calls += 1
        if self.fail:
            raise AnalysisError("Лимит Groq")
        return ProfileReview(strengths=STRENGTHS, checks=[], suggestions=[])


def completed(tmp_path):
    store = ProfileStore(tmp_path / "strengths.sqlite3")
    analyzer, reviewer = Analyzer(), Reviewer()
    interview = Interview(store, analyzer, reviewer=reviewer)
    interview.handle(1, "ML инженер")
    interview.handle(1, "; ".join(f.evidence for f in FACTS))
    interview.handle(1, "/continue")
    return interview, store, analyzer, reviewer


def make_legacy(store, analyzer, reviewer):
    with SqliteSaver.from_conn_string(str(store.path)) as saver:
        graph = build_graph(analyzer, saver, reviewer=reviewer)
        graph.update_state({"configurable": {"thread_id": "1"}},
                           {"review": {"highlight_fact_ids": [0, 1, 2], "checks": [], "suggestions": []}},
                           as_node="review_profile")


def test_authored_strengths_match_both_views_and_survive_restart(tmp_path):
    interview, store, analyzer, reviewer = completed(tmp_path)
    brief, full = interview.handle(1, "/review"), interview.handle(1, "/review_full")
    for text in (brief, full):
        for strength in STRENGTHS:
            assert strength["text"] in text
        assert "• Артак" not in text
        assert "учусь YSU" not in text
        assert "• изучаю Python" not in text
    restarted = Interview(ProfileStore(store.path), analyzer, reviewer=reviewer)
    assert restarted.handle(1, "/review") == brief
    assert restarted.handle(1, "/review_full") == full
    assert reviewer.calls == analyzer.calls == 1
    with sqlite3.connect(store.path) as db:
        saved = json.loads(db.execute("SELECT review_json FROM profiles WHERE user_id=1").fetchone()[0])
    assert saved["strengths"] == STRENGTHS


@pytest.mark.parametrize("command", ["/review", "/review_full"])
def test_legacy_review_regenerated_once_without_losing_answers(tmp_path, command):
    interview, store, analyzer, reviewer = completed(tmp_path)
    before = interview.get_state(1)
    make_legacy(store, analyzer, reviewer)
    restarted = Interview(ProfileStore(store.path), analyzer, reviewer=reviewer)
    assert reviewer.calls == 1  # Инициализация и профиль не вызывают модель.
    assert restarted.handle(1, "/profile") == interview.handle(1, "/profile")
    assert STRENGTHS[0]["text"] in restarted.handle(1, command)
    restarted.handle(1, "/review_full")
    restarted.handle(1, "/review")
    assert reviewer.calls == 2
    after = restarted.get_state(1)
    assert before["turns"] == after["turns"]
    assert before["facts"] == after["facts"]
    assert analyzer.calls == 1


def test_failed_legacy_refresh_can_retry_without_raw_fact_fallback(tmp_path):
    interview, store, analyzer, reviewer = completed(tmp_path)
    make_legacy(store, analyzer, reviewer)
    reviewer.fail = True
    reply = interview.handle(1, "/review")
    assert "Продолжить" in reply
    assert "• Артак" not in reply
    reviewer.fail = False
    restarted = Interview(ProfileStore(store.path), analyzer, reviewer=reviewer)
    restarted.handle(1, "/go")
    assert STRENGTHS[0]["text"] in restarted.handle(1, "/review_full")
    assert len(restarted.get_state(1)["turns"]) == 1
    assert analyzer.calls == 1


@pytest.mark.parametrize("strength", [
    {"text": "Сильная сторона — Артак.", "source_fact_ids": [0]},
    {"text": "Сильная сторона — Артак и Python.", "source_fact_ids": [0, 2]},
    {"text": "Опыт разработки на Python.", "source_fact_ids": [2, 999]},
    {"text": "Опыт разработки на Python.", "source_fact_ids": [-1]},
    {"text": "учусь YSU факультет физики 3 курс", "source_fact_ids": [1]},
    {"text": "Окончил физический факультет YSU.", "source_fact_ids": [1]},
    {"text": "Имеет 10 лет опыта Python.", "source_fact_ids": [2]},
])
def test_invalid_strength_is_not_replaced_with_a_raw_quote(strength):
    review = clean_review(ProfileReview(strengths=[strength], checks=[], suggestions=[]),
                          [f.model_dump() for f in FACTS])
    assert review["strengths"] == []
    text = review_text({"review": review, "facts": [f.model_dump() for f in FACTS]})
    assert "Сильные стороны" not in text
    assert "учусь YSU" not in text


def test_empty_strengths_are_cached_and_not_regenerated(tmp_path):
    interview, store, analyzer, reviewer = completed(tmp_path)
    with SqliteSaver.from_conn_string(str(store.path)) as saver:
        graph = build_graph(analyzer, saver, reviewer=reviewer)
        graph.update_state({"configurable": {"thread_id": "1"}},
                           {"review": {"strengths": [], "checks": [], "suggestions": []}},
                           as_node="review_profile")
    interview.handle(1, "/review")
    interview.handle(1, "/review_full")
    assert reviewer.calls == 1
