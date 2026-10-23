"""Граф: приём ответа → интервьюер → выбор вопроса → ревьюер профиля.

Один вызов обрабатывает один ход; checkpoints сохраняют состояние между сообщениями.
"""

from langgraph.graph import END, START, StateGraph

from cv_assistant.llm import Analyzer
from cv_assistant.memory import repeat_only, answers_revision, clean_facts, merge_summaries, find_gaps, gap_id
from cv_assistant.models import Analysis, InterviewState, ProfileReview
from cv_assistant.reviewer import OfflineReviewer, Reviewer, clean_review
from cv_assistant.languages import extract_languages

ROLE_QUESTION = "🎯 Какую работу ты ищешь? Напиши желаемую должность."
ABOUT_QUESTION = (
    "💬 Расскажи немного о себе: чем занимаешься и что умеешь. Я уточню важные детали."
)
COMPLETED = "✅ Анкета сохранена. Профиль и рекомендации доступны в меню."


def initial_state(target_role: str = "", about: str = "", stage: str = "target_role") -> InterviewState:
    return {
        "target_role": target_role, "about": about, "stage": stage,
        "turns": ([{"topic": "about", "key": "overview", "question": ABOUT_QUESTION, "answer": about}] if about else []),
        "facts": [], "asked_topics": [], "asked_keys": [], "gaps": [], "review": None,
        "languages": [], "analysis_result": {}, "summaries": [], "analysis_revision": "",
        "question": ROLE_QUESTION if stage == "target_role" else ABOUT_QUESTION,
        "question_topic": "about", "question_key": "overview", "action": "answer",
        "incoming_text": "", "finish_requested": False,
        "needs_analysis": False, "reply": "", "completion_reason": "",
    }


def receive(state: InterviewState) -> dict:
    if state.get("action") == "resume":
        analyzed = state.get("analysis_revision") == answers_revision(state["turns"])
        # Совместимость с уже разобранными анкетами до появления номера версии.
        if not state.get("analysis_revision") and state.get("analysis_result"):
            analyzed = state["facts"] == clean_facts(state["facts"], state["turns"])
        return {"stage": "followup" if state["turns"] else "about",
                "needs_analysis": bool(state["turns"]) and (state["needs_analysis"] or not analyzed), "review": None,
                "asked_keys": [], "asked_topics": [], "completion_reason": "",
                "question": ABOUT_QUESTION, "question_topic": "about", "question_key": "overview",
                "reply": ABOUT_QUESTION}
    if state["finish_requested"]:
        return {"stage": "complete", "completion_reason": "user", "needs_analysis": False,
                "reply": COMPLETED, "review": None}
    text = state["incoming_text"]
    if state["stage"] == "target_role":
        return {"target_role": text, "stage": "about", "question": ABOUT_QUESTION,
                "question_topic": "about", "question_key": "overview",
                "reply": ABOUT_QUESTION, "needs_analysis": False, "review": None}
    turn = {"topic": state["question_topic"], "key": state.get("question_key", "overview"),
            "question": state["question"], "answer": text}
    return {"turns": [*state["turns"], turn], "needs_analysis": True, "review": None,
            "about": text if state["stage"] == "about" else state["about"]}


def build_graph(analyzer: Analyzer, checkpointer, max_questions: int = 5, reviewer: Reviewer | None = None):
    reviewer = reviewer or OfflineReviewer()

    def analyze(state: InterviewState) -> dict:
        analysis = Analysis.model_validate(analyzer.analyze(state["target_role"], state["turns"]))
        return {"analysis_result": analysis.model_dump()}

    def validate_answers(state: InterviewState) -> dict:
        analysis = Analysis.model_validate(state["analysis_result"])
        facts = clean_facts([fact.model_dump() for fact in analysis.facts], state["turns"])
        # Замечание о повторе — не новые биографические сведения.
        if state["turns"] and repeat_only(state["turns"][-1]["answer"]):
            facts = clean_facts(state["facts"], state["turns"])
        gaps = find_gaps(state["target_role"], facts, [gap.model_dump() for gap in analysis.gaps], state["turns"])
        languages = extract_languages(state["turns"], [item.model_dump() for item in analysis.languages])
        summaries = merge_summaries(state.get("summaries", []), [s.model_dump() for s in analysis.summaries],
                                    state["facts"], facts, state["turns"])
        return {"facts": facts, "gaps": gaps, "languages": languages, "summaries": summaries,
                "needs_analysis": False, "analysis_revision": answers_revision(state["turns"])}

    def choose_next(state: InterviewState) -> dict:
        gaps = state.get("gaps", find_gaps(state["target_role"],
                         clean_facts(state["facts"], state["turns"]), [], state["turns"]))
        asked = state.get("asked_keys", [f"{topic}:overview" for topic in state["asked_topics"]])
        if not gaps:
            if state.get("action") == "resume":
                question = "💬 Что хочешь добавить или исправить в анкете? Напиши одним сообщением."
                return {"stage": "followup", "question": question, "question_topic": "about",
                        "question_key": "overview", "reply": question}
            return {"stage": "complete", "completion_reason": "enough",
                    "reply": COMPLETED}
        if len(asked) >= max_questions:
            return {"stage": "complete", "completion_reason": "limit",
                    "reply": "✅ Анкета сохранена. Ещё есть пробелы — можно дополнить её позже через «Продолжить»."}
        # Отвеченные вопросы отсекаются по всей истории в find_gaps, включая прошлые раунды.
        available = [gap for gap in gaps if gap_id(gap) not in asked]
        available.sort(key=lambda gap: asked.count(gap_id(gap)))
        if not available:
            return {"stage": "complete", "completion_reason": "gaps",
                    "reply": "✅ Ответы сохранены. Оставшиеся детали можно добавить позже."}
        gap = available[0]
        return {"stage": "followup", "question": gap["question"], "question_topic": gap["topic"],
                "question_key": gap["key"], "asked_keys": [*asked, gap_id(gap)],
                "asked_topics": [*state["asked_topics"], gap["topic"]],
                "reply": "💬 " + gap["question"]}

    def review_profile(state: InterviewState) -> dict:
        facts = clean_facts(state["facts"], state["turns"])
        gaps = find_gaps(state["target_role"], facts, state.get("gaps", []), state["turns"])
        review = ProfileReview.model_validate(reviewer.review(state["target_role"], facts, gaps))
        data = {"review": clean_review(review, facts), "facts": facts, "gaps": gaps}
        return {**data, "reply": state["reply"]}

    def after_receive(state: InterviewState) -> str:
        if state["needs_analysis"]:
            return "analyze"
        if state.get("action") == "resume" and state["turns"]:
            return "choose_next"
        return "review_profile" if state["stage"] == "complete" else END

    builder = StateGraph(InterviewState)
    builder.add_node("receive", receive)
    builder.add_node("analyze", analyze)
    builder.add_node("validate_answers", validate_answers)
    builder.add_node("choose_next", choose_next)
    builder.add_node("review_profile", review_profile)
    builder.add_edge(START, "receive")
    builder.add_conditional_edges("receive", after_receive)
    builder.add_edge("analyze", "validate_answers")
    builder.add_edge("validate_answers", "choose_next")
    builder.add_conditional_edges("choose_next", lambda state: "review_profile" if state["stage"] == "complete" else END)
    builder.add_edge("review_profile", END)
    return builder.compile(checkpointer=checkpointer)
