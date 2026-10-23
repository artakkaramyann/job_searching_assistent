import pytest

from cv_assistant.interview import Interview
from cv_assistant.languages import extract_languages
from cv_assistant.memory import clean_facts, clean_summaries, find_gaps
from cv_assistant.models import Analysis, Fact, Gap, SectionSummary
from cv_assistant.storage import ProfileStore


def turn(answer, topic="about", key="overview"):
    return {"answer": answer, "topic": topic, "key": key, "question": "Расскажи"}


@pytest.mark.parametrize("answer, expected", [
    ("армянский русский английский все на уровне максимальном",
     {"Армянский": "максимальный (самооценка)", "Русский": "максимальный (самооценка)", "Английский": "максимальный (самооценка)"}),
    ("Армянский родной, русский свободно, английский B2",
     {"Армянский": "родной", "Русский": "свободно", "Английский": "B2"}),
    ("Русский и английский", {"Русский": None, "Английский": None}),
    ("Русский, английский B1", {"Русский": None, "Английский": "B1"}),
])
def test_languages_and_levels_are_separate_and_not_invented(answer, expected):
    result = extract_languages([turn(answer)], [{"name": "Английский", "level": "C2", "evidence": answer}])
    assert {item["name"]: item["level"] for item in result} == expected


def test_later_language_level_updates_one_language():
    result = extract_languages([turn("Русский родной, английский B1"), turn("Английский B2")])
    assert {item["name"]: item["level"] for item in result} == {"Русский": "родной", "Английский": "B2"}


def test_llm_language_casing_and_aliases_do_not_duplicate_entries():
    answer = "армянский русский английский все на уровне максимальном"
    proposed = [{"name": name, "level": "максимальном", "evidence": answer}
                for name in ["армянский", "русский", "английский", "Armenian"]]
    result = extract_languages([turn(answer)], proposed)
    assert len(result) == 3
    assert all(item["level"] == "максимальный (самооценка)" for item in result)


def test_chef_uses_ai_question_and_preserves_already_known_experience(tmp_path):
    story = "Я повар из шаурмичной в Ереване уже 10 лет готовлю шаурму"
    question = "Какие десерты ты готовил сам?"
    facts = [Fact(topic="experience", text=story, evidence=story)]
    proposed = [Gap(topic="skills", key="professional_tools", question=question),
                Gap(topic="skills", key="programming_languages", question="Какие языки программирования знаешь?"),
                Gap(topic="projects", key="overview", question="Какие учебные проекты у тебя есть?")]

    class Model:
        def analyze(self, target_role, turns):
            return Analysis(facts=facts, gaps=proposed)

    interview = Interview(ProfileStore(tmp_path / "db.sqlite3"), Model())
    interview.handle(1, "Повар кондитер")
    assert interview.handle(1, story) == "💬 " + question
    state = interview.get_state(1)
    assert state["facts"][0]["text"] == story
    assert not any(g["topic"] in {"experience", "projects"} for g in state["gaps"])
    assert not any(g["key"] == "programming_languages" for g in state["gaps"])


def test_ai_overview_has_priority_over_generic_question():
    question = "С каким кухонным оборудованием ты работаешь?"
    gaps = find_gaps("Повар", [], [Gap(topic="skills", key="overview", question=question).model_dump()], [])
    assert gaps[0]["question"] == question


def test_explicit_shared_language_level_does_not_trigger_repeated_question():
    turns = [turn("армянский русский английский все на уровне максимальном")]
    gaps = find_gaps("Повар", [], [Gap(topic="languages", key="language_levels", question="Какой уровень?").model_dump()], turns)
    assert all(g["topic"] != "languages" for g in gaps)


def test_screenshot_four_languages_native_like_and_typo():
    turns = [turn("Армянский, русский, английский и французкий на всех разговариваю как носитель")]
    languages = extract_languages(turns)
    assert {x["name"] for x in languages} == {"Армянский", "Русский", "Английский", "Французский"}
    assert all(x["level"] == "как носитель (самооценка)" for x in languages)
    assert not any(g["topic"] == "languages" for g in find_gaps("Физик", [], [], turns))


def test_ambiguous_shared_level_is_clarified_and_confirmation_resolves_it():
    turns = [turn("Армянский, русский, английский и французкий знаю как носитель")]
    assert all(x["level"] is None for x in extract_languages(turns))
    question = find_gaps("Физик", [], [], turns)[0]
    assert question["key"] == "language_levels"
    assert "Правильно понимаю" in question["question"]
    assert "Французский" in question["question"]
    turns.append({**turn("Да, на всех", "languages", "language_levels"), "question": question["question"]})
    assert all(x["level"] == "как носитель (самооценка)" for x in extract_languages(turns))
    assert not any(g["topic"] == "languages" for g in find_gaps("Физик", [], [], turns))


def test_missing_level_is_not_guessed_and_decline_not_repeated():
    turns = [turn("Французский")]
    gap = find_gaps("Физик", [], [], turns)[0]
    assert gap["key"] == "language_levels"
    turns.append(turn("Не знаю", "languages", "language_levels"))
    assert extract_languages(turns)[0]["level"] is None
    assert not any(g["topic"] == "languages" for g in find_gaps("Физик", [], [gap], turns))


def test_rephrased_question_and_cross_topic_equipment_question_are_not_repeated():
    turns = [{**turn("Я был теоретиком, оборудование не использовал", "skills", "professional_tools"),
              "question": "Какие инструменты и оборудование ты использовал?"},
             turn("Пытался объяснить результаты эксперимента", "projects", "project_details")]
    proposed = [Gap(topic="skills", key="professional_tools", question="С какими приборами работал?"),
                Gap(topic="experience", key="experience_details", question="Что за оборудование применял в лаборатории?"),
                Gap(topic="projects", key="project_details", question="Что именно делал в этом проекте?")]
    gaps = find_gaps("Физик", [], [g.model_dump() for g in proposed], turns)
    assert not any(g["key"] in {"professional_tools", "experience_details", "project_details"} for g in gaps)


@pytest.mark.parametrize("answer", ["Я уже отвечал это единственный проект",
                                    "я отвечал уже это единственный проект", "тоже отвечал", "отвечал"])
def test_already_answered_is_not_a_biographical_fact(answer):
    turns = [turn(answer, "projects", "project_details")]
    facts = [Fact(topic="projects", text=turns[0]["answer"], evidence=turns[0]["answer"]).model_dump()]
    assert clean_facts(facts, turns) == []
    gaps = find_gaps("Developer", [], [], turns)
    assert not any(g["topic"] == "projects" for g in gaps)


def test_summary_is_authored_by_agent_and_raw_history_preserved(tmp_path):
    story = "Я учусь в Ереване YSU факультет физики 3 курс"
    summary = "Учится на 3-м курсе физического факультета YSU в Ереване."

    class Model:
        def analyze(self, target_role, turns):
            return Analysis(facts=[Fact(topic="education", text=story, evidence=story)], gaps=[],
                            summaries=[SectionSummary(topic="education", text=summary, evidence=[story])])

    interview = Interview(ProfileStore(tmp_path / "db.sqlite3"), Model())
    interview.handle(1, "Физик")
    interview.handle(1, story)
    profile = interview.handle(1, "/profile")
    assert "Образование: " + summary in profile
    assert story not in profile
    assert story in interview.handle(1, "/history")
    assert interview.get_state(1)["facts"][0]["evidence"] == story
    assert summary in Interview(ProfileStore(interview.store.path), Model()).handle(1, "/profile")


def test_summary_with_wrong_source_or_new_numbers_is_not_shown():
    facts = [Fact(topic="education", text="Учусь на 3 курсе YSU", evidence="Учусь на 3 курсе YSU").model_dump()]
    assert clean_summaries([
        {"topic": "education", "text": "Окончил Гарвард.", "evidence": ["Окончил Гарвард"]},
        {"topic": "education", "text": "Учится на 4 курсе YSU.", "evidence": ["Учусь на 3 курсе YSU"]},
        {"topic": "experience", "text": "Учится на 3 курсе.", "evidence": ["Учусь на 3 курсе YSU"]},
    ], facts) == []


@pytest.mark.parametrize("answer", ["Оборудование не использовал, работал только с теорией.",
                                    "Не использовал никаких приборов.", "Не работала с оборудованием."])
def test_no_equipment_in_initial_story_prevents_rephrased_questions(answer):
    proposed = [Gap(topic="skills", key="professional_tools", question="Какими приборами пользовался?"),
                Gap(topic="experience", key="experience_details", question="Какое оборудование применял?")]
    gaps = find_gaps("Физик", [], [g.model_dump() for g in proposed], [turn(answer)])
    assert not any(g["key"] in {"professional_tools", "experience_details"} for g in gaps)


def test_summary_cannot_turn_attempt_into_success_or_student_into_graduate():
    evidence = "Пытался теоретически объяснить результаты оптического эксперимента"
    facts = [Fact(topic="projects", text="теоретически объяснить результаты оптического эксперимента",
                  evidence=evidence).model_dump()]
    facts = clean_facts(facts, [turn(evidence, "projects")])
    assert facts[0]["text"] == evidence
    assert clean_summaries([{"topic": "projects", "text": "Теоретически объяснил результаты эксперимента.",
                             "evidence": ["теоретически объяснить результаты оптического эксперимента"]}], facts) == []
    safe = {"topic": "projects", "text": "Пытался теоретически объяснить результаты эксперимента.", "evidence": [evidence]}
    assert clean_summaries([safe], facts) == [safe]
    facts = [Fact(topic="education", text="Учусь в YSU", evidence="Учусь в YSU").model_dump()]
    assert clean_summaries([{"topic": "education", "text": "Окончил YSU.", "evidence": ["Учусь в YSU"]}], facts) == []


def test_resume_keeps_answered_question_history(tmp_path):
    class Model:
        def analyze(self, target_role, turns):
            return Analysis(facts=[], gaps=[Gap(topic="skills", key="professional_tools",
                                               question="Какие инструменты использовал?")])

    interview = Interview(ProfileStore(tmp_path / "db.sqlite3"), Model())
    interview.handle(1, "Физик")
    assert "инструменты" in interview.handle(1, "Изучаю оптику")
    assert "инструменты" not in interview.handle(1, "Я был теоретиком")
    interview.handle(1, "/continue")
    assert "инструменты" not in interview.handle(1, "/resume")
    assert len(interview.get_state(1)["turns"]) == 2
