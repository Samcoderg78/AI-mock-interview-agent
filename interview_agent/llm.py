"""Model factory. Any provider works through init_chat_model; MOCK_LLM=1 gives an offline fake."""
import os
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages.utils import convert_to_messages
from langchain_core.runnables import Runnable, RunnableLambda

from .schemas import (Evaluation, ExperienceItem, Feedback, InterviewPlan, JobProfile, PlanTopic, Question,
                      ReportNarrative, ResumeProfile)


def get_model(temperature: float = 0.4):
    return init_chat_model(os.getenv("MODEL", "openai:gpt-4o-mini"), temperature=temperature)


def structured(schema: type, temperature: float = 0.3) -> Runnable:
    """Runnable: messages -> validated Pydantic object. Retries transient/validation failures."""
    if os.getenv("MOCK_LLM") == "1":
        return RunnableLambda(lambda msgs: _fake(schema, msgs)).with_retry(stop_after_attempt=2)
    return get_model(temperature).with_structured_output(schema).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )


# ---------------------------------------------------------------- offline fake (tests / demo)
def _field(text: str, key: str, default: str = "") -> str:
    m = re.search(rf"^{key}:\s*(.*)$", text, re.M)
    return m.group(1).strip() if m else default


def _fake(schema: type, msgs: Any):
    text = "\n".join(str(m.content) for m in convert_to_messages(msgs))
    if schema is Question:
        topic, diff, kind = _field(text, "TOPIC", "General"), _field(text, "DIFFICULTY", "medium"), _field(text, "KIND", "main")
        return Question(
            text=f"({kind}, {diff}) Explain the key ideas of: {topic}.",
            topic=topic, difficulty=diff, kind=kind, style=_field(text, "STYLE", "technical"),
            ideal_points=["definition", "example", "trade-offs"],
        )
    if schema is Evaluation:
        tail = text.rsplit("<candidate_answer>", 1)[-1].split("</candidate_answer>")[0]
        words = len(tail.split()) if "<candidate_answer>" in text else 0
        s = min(10, words // 4)
        return Evaluation(correctness=s, depth=max(0, s - 1), clarity=min(10, s + 1),
                          strengths=["clear start"] if s >= 5 else [],
                          gaps=["add examples"] if s < 8 else [],
                          missed_concepts=["trade-offs"] if s < 7 else [],
                          needs_follow_up=s < 6)
    if schema is ResumeProfile:
        body = text.rsplit("<resume>", 1)[-1]
        exps = [ExperienceItem(name=m.group(1).strip(), description=(m.group(2) or "").strip(), skills=["Teamwork"])
                for m in re.finditer(r"^(?:Project|Experience):\s*(.+?)(?:\s+-\s+(.*))?$", body, re.M)]
        return ResumeProfile(summary="Offline resume summary", skills=["Communication", "Excel"], experiences=exps)
    if schema is JobProfile:
        body = text.rsplit("<job_description>", 1)[-1]
        reqs = [m.group(1).strip() for m in re.finditer(r"^Requirement:\s*(.+)$", body, re.M)]
        return JobProfile(title=_field(body, "Title", ""), requirements=reqs, summary="Offline job summary")
    if schema is InterviewPlan:
        setting = _field(text, "LEVEL_SETTING", "mid")
        focus = [x.strip() for x in _field(text, "FOCUS_TOPICS", "none").split(";") if x.strip() and x.strip() != "none"]
        reqs = [x.strip() for x in _field(text, "JOB_REQUIREMENTS", "none").split(";") if x.strip() and x.strip() != "none"]
        names = reqs or focus or ["Core role skills", "Problem solving", "Communication"]
        return InterviewPlan(role_title=_field(text, "ROLE", "Professional").replace("not given", "Professional"),
                             level="mid" if setting == "auto" else setting,
                             topics=[PlanTopic(name=n, style="behavioral" if "Communication" in n else "technical")
                                     for n in names])
    if schema is Feedback:
        return Feedback(summary="Reasonable attempt; keep going.", improvement_tip="Add a concrete example.",
                        model_answer="A strong answer defines the concept, shows an example and names trade-offs.")
    if schema is ReportNarrative:
        return ReportNarrative(summary="Offline demo report.", role_fit_summary="Offline fit summary.", strengths=["Consistent effort"],
                               improvements=["Go deeper on trade-offs"],
                               study_plan=["Review weak topics for 30 minutes daily"],
                               readiness="almost_ready")
    raise ValueError(f"No fake for {schema}")
