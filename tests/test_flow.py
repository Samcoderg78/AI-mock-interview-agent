import os
import pytest

os.environ["MOCK_LLM"] = "1"

from interview_agent.graph import build_graph                                                  # noqa: E402
from interview_agent.nodes import (build_agenda, merge_topics, plan_next, shift_difficulty,     # noqa: E402
                                   wants_follow_up)
from interview_agent.cli import run_interview                                                   # noqa: E402
from interview_agent.memory import load_profile                                                 # noqa: E402

LONG = "word " * 40      # fake evaluator scores by length: long answer -> high score
SHORT = "idk"
JD = "Title: Marketing Manager\nRequirement: Campaign analytics\nRequirement: Stakeholder management\nRequirement: Budget planning"
RESUME = ("Skills: SEO\nExperience: Led Q3 launch campaign - grew leads 40 percent\n"
          "Project: Brand refresh - new identity\nProject: Third - ignored")


@pytest.fixture(autouse=True)
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("MOCK_INTERVIEW_HOME", str(tmp_path))


def drive(answers, **kw):
    it = iter(answers)
    events = []
    config = {"candidate_name": "Test User", "level": "mid", **kw}
    final = run_interview(build_graph(), config, "t-" + str(id(answers)),
                          ask=lambda _: next(it), out=lambda n, u: events.append((n, u)))
    return final, events


def questions(events):
    return [u["question"] for n, u in events if n == "generate_question"]


def update(events, node):
    return next(u for n, u in events if n == node)


def test_job_description_mode_plans_topics_from_the_posting_and_ends_by_itself():
    final, events = drive([LONG] * 3, job_description=JD)
    names = [n for n, _ in events]
    assert names[:8] == ["interview_configuration", "resume_analysis", "job_analysis", "interview_planner",
                         "interview_agent", "generate_question", "candidate_answer", "answer_evaluator"]
    assert [q["topic"] for q in questions(events)] == ["Campaign analytics", "Stakeholder management", "Budget planning"]
    assert update(events, "interview_planner")["config"]["role"] == "Marketing Manager"   # role from the JD
    assert "Overall score" in final["report"] and "Fit for the role" in final["report"]


def test_works_for_any_role_with_no_python_assumptions():
    _, events = drive([LONG] * 3, role="Nurse")          # role only: planner falls back to generic topics
    assert len(questions(events)) == 3
    assert "Python" not in update(events, "interview_agent")["system_prompt"]
    assert "Nurse" in update(events, "interview_agent")["system_prompt"]


def test_level_auto_is_resolved():
    _, events = drive([LONG] * 3, role="Nurse", level="auto")
    assert update(events, "interview_planner")["config"]["level"] == "mid"
    assert questions(events)[0]["difficulty"] == "medium"


def test_focus_topics_come_first_and_resume_adds_deep_dives():
    _, events = drive([LONG] * 6, job_description=JD, resume_text=RESUME, focus_topics=["Public speaking"])
    qs = questions(events)
    assert qs[0]["topic"] == "Public speaking"
    resume_qs = [q for q in qs if q["kind"] == "resume"]
    assert [q["topic"] for q in resume_qs] == ["From your resume: Led Q3 launch campaign",
                                               "From your resume: Brand refresh"]   # max 2; the third is ignored
    assert qs[1]["kind"] == "resume"                     # woven in, not all at the start


def test_internal_topic_cap_limits_long_job_descriptions():
    jd = "\n".join(f"Requirement: Skill {i}" for i in range(9))
    _, events = drive([LONG] * 6, job_description=jd)
    assert len(questions(events)) == 6                   # max_topics = 6


def test_weak_answer_adds_one_follow_up_then_continues():
    _, events = drive([SHORT, LONG, LONG, LONG], job_description=JD)
    qs = questions(events)
    assert len(qs) == 4
    assert qs[1]["kind"] == "follow_up" and qs[1]["topic"] == qs[0]["topic"]
    assert qs[2]["kind"] == "main" and qs[2]["topic"] != qs[0]["topic"]


def test_strong_answers_raise_difficulty():
    _, events = drive([LONG] * 3, job_description=JD)
    assert [q["difficulty"] for q in questions(events)] == ["medium", "hard", "hard"]


def test_quit_skip_and_safety_cap():
    final, _ = drive([LONG, "/quit"], job_description=JD)
    assert "ended early" in final["report"]
    _, events = drive(["/skip", LONG, LONG], job_description=JD)
    assert update(events, "structured_evaluation")["evaluation"]["overall"] == 0 and len(questions(events)) == 3
    _, events = drive([LONG] * 5, job_description=JD, max_questions=2)
    assert len(questions(events)) == 2


def test_requires_some_input():
    with pytest.raises(ValueError):
        drive([LONG])


def test_memory_saved_without_resume_deep_dive_topics():
    drive([SHORT, SHORT, LONG, LONG, LONG, LONG], job_description=JD, resume_text=RESUME)
    prof = load_profile("Test User")
    assert prof["sessions"] == 1 and prof["weak"]
    assert not any(t.startswith("From your resume") for t in prof["topics"])


def test_pure_helpers():
    assert shift_difficulty("easy", -1) == "easy" and shift_difficulty("hard", 1) == "hard"
    topics = merge_topics([{"name": "A"}, {"name": "b"}, {"name": "C"}], ["B", "Z"], max_topics=4)
    assert [t["name"] for t in topics] == ["B", "Z", "A", "C"]          # focus first, case-insensitive dedupe
    agenda = build_agenda([{"name": "A"}, {"name": "B"}, {"name": "C"}], ["P1", "P2", "P3"], max_resume=2)
    assert [i["topic"] for i in agenda] == ["A", "From your resume: P1", "B", "From your resume: P2", "C"]
    weak = {"question": {"topic": "A", "kind": "main", "difficulty": "medium", "style": "technical"},
            "evaluation": {"overall": 3, "needs_follow_up": True}}
    assert wants_follow_up({"max_follow_ups_per_topic": 1}, [weak])
    assert not wants_follow_up({"max_follow_ups_per_topic": 0}, [weak])
    plan, rest = plan_next({"max_follow_ups_per_topic": 0}, [weak], [{"topic": "B", "kind": "main", "style": "case"}])
    assert plan["topic"] == "B" and plan["difficulty"] == "easy" and plan["style"] == "case" and rest == []
