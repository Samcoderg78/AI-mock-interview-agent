import operator
from typing import Annotated, Any
from typing_extensions import TypedDict


class InterviewState(TypedDict, total=False):
    # inputs / configuration
    config: dict[str, Any]
    system_prompt: str
    # analysis and planning
    prior_profile: dict[str, Any]   # skills remembered from earlier sessions
    resume_profile: dict[str, Any]  # facts extracted from the resume (may be empty)
    job_profile: dict[str, Any]     # facts extracted from the job description (may be empty)
    interview_plan: dict[str, Any]  # role, level and topics chosen by the planner
    agenda: list[dict[str, Any]]    # planned main/resume questions still to ask
    # current turn
    next_plan: dict[str, Any]       # {"topic", "difficulty", "kind", "style"} chosen for the next question
    question: dict[str, Any]
    answer: str
    evaluation: dict[str, Any]
    feedback: dict[str, Any]
    # Interview Memory (reducer appends one record per completed turn)
    turns: Annotated[list[dict[str, Any]], operator.add]
    asked: int
    ended_early: bool
    # output
    report: str
    report_path: str
