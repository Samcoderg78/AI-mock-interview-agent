"""Terminal interface, e.g.:
python -m interview_agent.cli --name Asha --resume resume.pdf --jd job.txt
python -m interview_agent.cli --role "Marketing Manager" --focus "campaign analytics" "stakeholder management"
"""
import argparse
import uuid

from dotenv import load_dotenv
from langgraph.types import Command

from .graph import build_graph
from .resume import load_resume


def show(node: str, update: dict) -> None:
    if node == "resume_analysis" and update["resume_profile"]:
        r = update["resume_profile"]
        print(f"Resume read: {len(r['experiences'])} experience item(s), {len(r['skills'])} skills detected.")
    elif node == "job_analysis" and update["job_profile"]:
        j = update["job_profile"]
        print(f"Job description read: {j['title'] or 'untitled role'}, {len(j['requirements'])} requirements found.")
    elif node == "interview_planner":
        pl = update["interview_plan"]
        print(f"\nInterview plan for a {pl['level']}-level {pl['role']}:")
        for t in pl["topics"]:
            print(f"  - {t['name']} ({t['style']})")
    elif node == "generate_question":
        q = update["question"]
        tag = {"follow_up": " | follow-up", "resume": " | from your resume"}.get(q["kind"], "")
        print(f"\n[{q['topic']} | {q['style']} | {q['difficulty']}{tag}]")
    elif node == "structured_evaluation":
        ev = update["evaluation"]
        print(f"\nScore: {ev['overall']}/10  (correctness {ev['correctness']}, depth {ev['depth']}, clarity {ev['clarity']})")
    elif node == "feedback_generator":
        fb = update["feedback"]
        print(f"Feedback: {fb['summary']}\nTip: {fb['improvement_tip']}\nModel answer: {fb['model_answer']}")
    elif node == "final_interview_report":
        print("\n" + "=" * 70 + "\n" + update["report"] + f"\n\nSaved to {update['report_path']}")


def run_interview(graph, config: dict, thread_id: str, ask=input, out=show) -> dict:
    """Drive the graph: stream node updates, and resume each interrupt with the candidate's answer."""
    cfg = {"configurable": {"thread_id": thread_id}}
    payload = {"config": config}
    final: dict = {}
    while True:
        pending = None
        for chunk in graph.stream(payload, cfg, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    pending = update[0].value
                else:
                    out(node, update)
                    if node == "final_interview_report":
                        final = update
        if pending is None:
            return final
        more = f"about {pending['remaining']} more planned" if pending["remaining"] else "last planned question"
        print(f"\nQuestion {pending['number']} ({more}): {pending['question']}")
        print("(type your answer; /skip to skip, /quit to end the interview now)")
        payload = Command(resume=ask("> "))


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="AI-powered mock interview for any role")
    p.add_argument("--name", default="Candidate")
    p.add_argument("--role", default="", help="job title (optional if you give a job description)")
    p.add_argument("--level", choices=["auto", "junior", "mid", "senior"], default="auto")
    p.add_argument("--jd", help="path to a job description (.pdf or .txt)")
    p.add_argument("--resume", help="path to your resume (.pdf or .txt)")
    p.add_argument("--focus", nargs="*", default=[], help="topics you want to practice")
    a = p.parse_args()
    config = {"candidate_name": a.name, "role": a.role, "level": a.level, "focus_topics": a.focus,
              "job_description": load_resume(a.jd) if a.jd else "",
              "resume_text": load_resume(a.resume) if a.resume else ""}
    print(f"Starting mock interview for {a.name}. Good luck!")
    run_interview(build_graph(), config, thread_id=str(uuid.uuid4()))


if __name__ == "__main__":
    main()
