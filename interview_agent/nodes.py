"""Graph nodes. Names mirror the architecture diagram, plus Resume/Job Analysis and Interview Planner."""
from datetime import datetime
from typing import Literal

from langgraph.types import interrupt

from . import memory, prompts
from .llm import structured
from .report import compute_stats, render_report, topic_scores
from .schemas import (Evaluation, Feedback, InterviewConfig, InterviewPlan, JobProfile, Question,
                      ReportNarrative, ResumeProfile)
from .state import InterviewState

LADDER = ["easy", "medium", "hard"]
LEVEL_START = {"junior": "easy", "mid": "medium", "senior": "hard"}


# ------------------------------------------------------------------ pure helpers (unit-tested)
def shift_difficulty(current: str, delta: int) -> str:
    return LADDER[max(0, min(len(LADDER) - 1, LADDER.index(current) + delta))]


def merge_topics(llm_topics: list[dict], focus: list[str], max_topics: int) -> list[dict]:
    """Candidate focus topics always come first; duplicates removed; capped at max_topics."""
    seen, out = set(), []
    for t in [{"name": f, "style": "technical", "rationale": "Requested by the candidate"} for f in focus] + llm_topics:
        key = t["name"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out[:max_topics]


def build_agenda(topics: list[dict], experiences: list[str], max_resume: int) -> list[dict]:
    """Planned questions. Resume deep-dives are woven in (never all at the start)."""
    items = [{"topic": t["name"], "kind": "main", "style": t.get("style", "technical")} for t in topics]
    for k, name in enumerate(experiences[:max_resume]):
        items.insert(min(len(items), 1 + 2 * k),
                     {"topic": prompts.RESUME_PREFIX + name, "kind": "resume", "style": "behavioral"})
    return items


def wants_follow_up(cfg: dict, turns: list[dict]) -> bool:
    """A weak answer with a revealing gap earns ONE targeted follow-up on the same topic."""
    if not turns:
        return False
    q, ev = turns[-1]["question"], turns[-1]["evaluation"]
    used = sum(1 for t in turns if t["question"]["kind"] == "follow_up" and t["question"]["topic"] == q["topic"])
    return (q["kind"] != "follow_up" and ev["needs_follow_up"] and ev["overall"] < 7
            and used < cfg["max_follow_ups_per_topic"])


def plan_next(cfg: dict, turns: list[dict], agenda: list[dict]) -> tuple[dict, list[dict]]:
    """Return (plan for the next question, remaining agenda)."""
    q, ev = turns[-1]["question"], turns[-1]["evaluation"]
    if wants_follow_up(cfg, turns):
        diff = shift_difficulty(q["difficulty"], -1 if ev["overall"] < 4 else 0)
        return {"topic": q["topic"], "difficulty": diff, "kind": "follow_up", "style": q.get("style", "technical")}, agenda
    delta = 1 if ev["overall"] >= 8 else -1 if ev["overall"] < 4 else 0
    return {**agenda[0], "difficulty": shift_difficulty(q["difficulty"], delta)}, agenda[1:]


# ------------------------------------------------------------------ nodes
def interview_configuration(state: InterviewState) -> dict:
    """Validate configuration and load what we remember about this candidate from past sessions."""
    cfg = InterviewConfig(**state.get("config", {}))
    if not cfg.has_input():
        raise ValueError("Provide at least one of: role, job description, resume, or focus topics.")
    return {"config": cfg.model_dump(), "prior_profile": memory.load_profile(cfg.candidate_name),
            "asked": 0, "ended_early": False}


def resume_analysis(state: InterviewState) -> dict:
    text = state["config"]["resume_text"].strip()
    if not text:
        return {"resume_profile": {}}
    return {"resume_profile": structured(ResumeProfile, 0.0).invoke(prompts.resume_messages(text)).model_dump()}


def job_analysis(state: InterviewState) -> dict:
    """Job Description mode: extract requirements and responsibilities (skipped without a job description)."""
    text = state["config"]["job_description"].strip()
    if not text:
        return {"job_profile": {}}
    return {"job_profile": structured(JobProfile, 0.0).invoke(prompts.job_messages(text)).model_dump()}


def interview_planner(state: InterviewState) -> dict:
    """Decide WHAT to ask: topics from the job, the resume, the candidate's focus and past weak areas."""
    cfg, job, resume = state["config"], state.get("job_profile", {}), state.get("resume_profile", {})
    plan: InterviewPlan = structured(InterviewPlan, 0.3).invoke(
        prompts.planner_messages(cfg, resume, job, state["prior_profile"]))
    role = cfg["role"] or plan.role_title or job.get("title") or "professional"
    level = plan.level if cfg["level"] == "auto" else cfg["level"]
    topics = merge_topics([t.model_dump() for t in plan.topics], cfg["focus_topics"], cfg["max_topics"]) or [
        {"name": f"Core skills for {role}", "style": "technical", "rationale": "fallback"}]
    agenda = build_agenda(topics, [e["name"] for e in resume.get("experiences", [])], cfg["max_resume_questions"])
    return {"config": {**cfg, "role": role, "level": level}, "agenda": agenda,
            "interview_plan": {"role": role, "level": level, "topics": topics}}


def interview_agent(state: InterviewState) -> dict:
    """Set the interviewer persona and take the first question from the agenda."""
    cfg = state["config"]
    first, *rest = state["agenda"]
    return {"system_prompt": prompts.interviewer_system_prompt(cfg), "agenda": rest,
            "next_plan": {**first, "difficulty": LEVEL_START[cfg["level"]]}}


def generate_question(state: InterviewState) -> dict:
    plan = state["next_plan"]
    q = structured(Question, 0.7).invoke(prompts.question_messages(
        state["system_prompt"], plan, state["config"], state.get("turns", []), state["prior_profile"],
        state.get("resume_profile", {}), state.get("job_profile", {})))
    # the plan (not the LLM) is the source of truth for metadata
    q = q.model_copy(update={k: plan[k] for k in ("topic", "difficulty", "kind", "style")})
    return {"question": q.model_dump()}


def candidate_answer(state: InterviewState) -> dict:
    """Human-in-the-loop: pause the graph until the candidate answers. No side effects before interrupt()."""
    q = state["question"]
    reply = interrupt({"number": state["asked"] + 1, "remaining": len(state.get("agenda", [])),
                       "topic": q["topic"], "difficulty": q["difficulty"], "kind": q["kind"],
                       "style": q["style"], "question": q["text"]})
    text = str(reply).strip()
    if text.lower() == "/quit":
        return {"answer": "", "ended_early": True}
    return {"answer": "" if text.lower() == "/skip" else text[:6000]}


def answer_evaluator(state: InterviewState) -> dict:
    if not state["answer"].strip():           # nothing to judge: skip the LLM call
        ev = Evaluation(correctness=0, depth=0, clarity=0, gaps=["No answer provided"])
    else:
        ev = structured(Evaluation, 0.0).invoke(
            prompts.evaluation_messages(state["system_prompt"], state["question"], state["answer"]))
    return {"evaluation": ev.model_dump()}


def structured_evaluation(state: InterviewState) -> dict:
    """Deterministic post-processing: weighted overall score and sane follow-up flag."""
    ev = Evaluation(**state["evaluation"])
    overall = round(0.5 * ev.correctness + 0.3 * ev.depth + 0.2 * ev.clarity, 1)
    follow = ev.needs_follow_up and overall < 7 and bool(state["answer"].strip())
    return {"evaluation": ev.model_copy(update={"overall": overall, "needs_follow_up": follow}).model_dump()}


def feedback_generator(state: InterviewState) -> dict:
    fb = structured(Feedback, 0.4).invoke(
        prompts.feedback_messages(state["question"], state["answer"], state["evaluation"]))
    return {"feedback": fb.model_dump()}


def interview_memory(state: InterviewState) -> dict:
    """Append this turn to the session memory (reducer-backed list in state)."""
    turn = {k: state[k] for k in ("question", "answer", "evaluation", "feedback")}
    return {"turns": [turn], "asked": state["asked"] + 1}


def interview_complete(state: InterviewState) -> Literal["adaptive_question_selection", "final_interview_report"]:
    """'Interview Complete?': done when the agenda is exhausted and no follow-up is pending.
    There is no user-set question count; length follows coverage. max_questions is only a safety cap."""
    cfg = state["config"]
    if state["asked"] >= cfg["max_questions"]:
        return "final_interview_report"
    if not state.get("agenda") and not wants_follow_up(cfg, state["turns"]):
        return "final_interview_report"
    return "adaptive_question_selection"


def adaptive_question_selection(state: InterviewState) -> dict:
    plan, agenda = plan_next(state["config"], state["turns"], state.get("agenda", []))
    return {"next_plan": plan, "agenda": agenda}


def final_interview_report(state: InterviewState) -> dict:
    cfg, turns, job = state["config"], state.get("turns", []), state.get("job_profile", {})
    stats = compute_stats(turns)
    narrative = (structured(ReportNarrative, 0.4).invoke(prompts.report_messages(cfg, stats, turns, job))
                 if turns else None)
    now = datetime.now()
    md = render_report(cfg, stats, narrative, turns, state.get("ended_early", False), now.strftime("%Y-%m-%d"))
    if turns:
        memory.update_profile(cfg["candidate_name"], topic_scores(turns))
    return {"report": md, "report_path": memory.save_report(cfg["candidate_name"], md, now.strftime("%Y%m%d-%H%M%S"))}
