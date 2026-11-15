"""Pure functions: statistics and Markdown rendering for the Final Interview Report."""
from collections import defaultdict
from statistics import mean

from .schemas import ReportNarrative

READINESS = {"not_ready": "Not ready yet", "almost_ready": "Almost ready", "ready": "Ready"}


def compute_stats(turns: list[dict]) -> dict:
    if not turns:
        return {"questions": 0, "overall": 0.0, "by_topic": {}, "by_difficulty": {}, "dimensions": {}, "skipped": 0}
    by_topic, by_diff = defaultdict(list), defaultdict(list)
    for t in turns:
        by_topic[t["question"]["topic"]].append(t["evaluation"]["overall"])
        by_diff[t["question"]["difficulty"]].append(t["evaluation"]["overall"])
    dims = {d: round(mean(t["evaluation"][d] for t in turns), 1) for d in ("correctness", "depth", "clarity")}
    return {
        "questions": len(turns),
        "overall": round(mean(t["evaluation"]["overall"] for t in turns), 1),
        "by_topic": {k: round(mean(v), 1) for k, v in by_topic.items()},
        "by_difficulty": {k: round(mean(v), 1) for k, v in by_diff.items()},
        "dimensions": dims,
        "skipped": sum(1 for t in turns if not t["answer"].strip()),
    }


def topic_scores(turns: list[dict]) -> dict[str, list[float]]:
    """Per-topic scores for the long-term profile (resume-project questions are not skill topics)."""
    out: dict[str, list[float]] = defaultdict(list)
    for t in turns:
        if t["question"]["kind"] == "resume":
            continue
        out[t["question"]["topic"]].append(t["evaluation"]["overall"])
    return dict(out)


def render_report(cfg: dict, stats: dict, narrative: ReportNarrative | None, turns: list[dict],
                  ended_early: bool, date: str) -> str:
    L = [f"# Mock Interview Report: {cfg['candidate_name']}",
         f"*{cfg['level'].title()} {cfg['role']} | {date}*", ""]
    if ended_early:
        L += ["> Interview ended early by the candidate.", ""]
    if not turns:
        return "\n".join(L + ["No questions were answered."])
    L += [f"## Overall score: {stats['overall']}/10", ""]
    if narrative:
        L += [f"**Readiness:** {READINESS[narrative.readiness]}", "", narrative.summary, "",
              f"**Fit for the role:** {narrative.role_fit_summary}", ""]
    L += ["## Score breakdown", "", "| Dimension | Score |", "|---|---|"]
    L += [f"| {k.title()} | {v}/10 |" for k, v in stats["dimensions"].items()]
    L += ["", "| Topic | Avg score |", "|---|---|"]
    L += [f"| {k} | {v}/10 |" for k, v in sorted(stats["by_topic"].items(), key=lambda kv: kv[1])]
    if narrative:
        L += ["", "## Strengths"] + [f"- {s}" for s in narrative.strengths]
        L += ["", "## Areas to improve"] + [f"- {s}" for s in narrative.improvements]
        L += ["", "## Study plan"] + [f"{i}. {s}" for i, s in enumerate(narrative.study_plan, 1)]
    L += ["", "## Question-by-question review"]
    for i, t in enumerate(turns, 1):
        q, ev, fb = t["question"], t["evaluation"], t["feedback"]
        L += ["", f"### Q{i}. {q['text']}",
              f"*{q['topic']} | {q['difficulty']} | {q['kind']} | score {ev['overall']}/10*", "",
              f"**Your answer:** {t['answer'] or '(skipped)'}", "",
              f"**Feedback:** {fb['summary']}", f"**Tip:** {fb['improvement_tip']}", "",
              f"**Model answer:** {fb['model_answer']}"]
    return "\n".join(L)
