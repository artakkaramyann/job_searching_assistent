"""Договор между графом и моделью: какие данные разрешено возвращать."""

from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

Topic = Literal["name", "experience", "skills", "projects", "education", "languages"]
REQUIRED_TOPICS: tuple[Topic, ...] = (
    "experience", "skills", "projects", "education", "languages"
)
LABELS = {
    "name": "Имя", "experience": "Опыт", "skills": "Навыки",
    "projects": "Проекты", "education": "Образование", "languages": "Языки",
}


class Fact(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: Topic
    text: str = Field(min_length=1, max_length=300)
    evidence: str = Field(min_length=1, max_length=300)


class Analysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    facts: list[Fact] = Field(max_length=24)
    gaps: list["Gap"] = Field(max_length=12)
    languages: list["LanguageSkill"] = Field(default_factory=list, max_length=20)
    summaries: list["SectionSummary"] = Field(default_factory=list, max_length=4)


class SectionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: Literal["experience", "skills", "projects", "education"]
    text: str = Field(min_length=1, max_length=500)
    evidence: list[str] = Field(min_length=1, max_length=8)


class LanguageSkill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    level: str | None
    evidence: str


class Gap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: Topic
    key: Literal["overview", "programming_languages", "libraries", "project_details",
                 "professional_tools", "experience_details", "education_details", "language_levels"]
    question: str = Field(min_length=1, max_length=500)


class StrengthSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=300)
    source_fact_ids: list[int] = Field(min_length=1, max_length=8)


class ProfileReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strengths: list[StrengthSummary] = Field(default_factory=list, max_length=4)
    checks: list[str] = Field(max_length=6)
    suggestions: list[str] = Field(max_length=6)


class Turn(TypedDict):
    topic: str
    question: str
    answer: str
    key: str


class InterviewState(TypedDict):
    target_role: str
    about: str
    stage: str
    turns: list[Turn]
    facts: list[dict]
    asked_topics: list[str]
    question: str
    question_topic: str
    incoming_text: str
    finish_requested: bool
    needs_analysis: bool
    reply: str
    completion_reason: str
    gaps: list[dict]
    asked_keys: list[str]
    question_key: str
    action: str
    review: dict | None
    languages: list[dict]
    analysis_result: dict
    summaries: list[dict]
    analysis_revision: str
