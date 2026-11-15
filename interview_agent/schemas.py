"""Pydantic schemas: the typed contracts between graph nodes and LLM calls."""
from typing import Literal
from pydantic import BaseModel, Field, field_validator

Difficulty = Literal["easy", "medium", "hard"]
Level = Literal["junior", "mid", "senior"]
LevelSetting = Literal["auto", "junior", "mid", "senior"]
Style = Literal["technical", "behavioral", "situational", "case"]


class InterviewConfig(BaseModel):
    """Interview Configuration node output. At least one of role / job_description / resume / focus is required."""
    candidate_name: str = "Candidate"
    role: str = ""                                   # optional: derived from the job description if empty
    level: LevelSetting = "auto"                     # "auto" infers seniority from the job description / resume
    job_description: str = ""                        # plain text of the job posting (optional)
    resume_text: str = ""                            # plain text of the resume (optional)
    focus_topics: list[str] = Field(default_factory=list, description="Topics the candidate wants to practice")
    max_resume_questions: int = Field(2, ge=0, le=4)
    max_follow_ups_per_topic: int = Field(1, ge=0, le=3)
    # Internal limits (not user settings): the interview ends when the planned agenda is covered.
    max_topics: int = Field(6, ge=1, le=10)
    max_questions: int = Field(20, ge=1, le=40)      # safety cap against runaway cost

    @field_validator("focus_topics")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        return [t.strip() for t in v if t and t.strip()]

    def has_input(self) -> bool:
        return bool(self.role.strip() or self.job_description.strip() or self.resume_text.strip() or self.focus_topics)


class ExperienceItem(BaseModel):
    name: str = Field(description="A project, achievement or role, e.g. 'Churn model at Acme'")
    description: str = ""
    skills: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    """Resume Analysis output: facts extracted from the resume, nothing invented."""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experiences: list[ExperienceItem] = Field(default_factory=list, description="Most substantial first, max 5")
    years_experience: float | None = None


class JobProfile(BaseModel):
    """Job Analysis output: what the employer is looking for."""
    title: str = ""
    seniority: Level | None = None
    domain: str = ""
    summary: str = ""
    requirements: list[str] = Field(default_factory=list, description="Must-have skills and qualifications")
    responsibilities: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)


class PlanTopic(BaseModel):
    name: str = Field(description="Short topic name, 2-6 words")
    style: Style = "technical"
    rationale: str = ""


class InterviewPlan(BaseModel):
    """Interview Planner output."""
    role_title: str
    level: Level
    topics: list[PlanTopic]


class Question(BaseModel):
    """Generate Question node output."""
    text: str = Field(description="The interview question, phrased as an interviewer would say it")
    topic: str
    difficulty: Difficulty
    kind: Literal["main", "follow_up", "resume"] = "main"
    style: Style = "technical"
    ideal_points: list[str] = Field(
        default_factory=list,
        description="3-5 key points a strong answer should cover (hidden from the candidate)",
    )


class Evaluation(BaseModel):
    """Answer Evaluator / Structured Evaluation output."""
    correctness: int = Field(ge=0, le=10, description="Accuracy and relevance of the content")
    depth: int = Field(ge=0, le=10, description="Depth, specifics, trade-offs, evidence")
    clarity: int = Field(ge=0, le=10, description="Structure and communication")
    overall: float = Field(default=0, ge=0, le=10)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    missed_concepts: list[str] = Field(default_factory=list)
    needs_follow_up: bool = False


class Feedback(BaseModel):
    """Feedback Generator output (shown to the candidate right after each answer)."""
    summary: str = Field(description="2-3 sentences of encouraging, specific feedback")
    improvement_tip: str = Field(description="One concrete thing to do better next time")
    model_answer: str = Field(description="A concise model answer or answer outline (3-6 sentences)")


class ReportNarrative(BaseModel):
    """LLM-written part of the Final Interview Report."""
    summary: str
    role_fit_summary: str = Field(description="How well the candidate fits the role, based only on the interview")
    strengths: list[str]
    improvements: list[str]
    study_plan: list[str] = Field(description="Concrete 1-2 week preparation actions")
    readiness: Literal["not_ready", "almost_ready", "ready"]
