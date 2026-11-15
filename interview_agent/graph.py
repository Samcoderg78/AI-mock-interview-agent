from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import nodes as n
from .state import InterviewState


def route_after_answer(state: InterviewState) -> str:
    return "final_interview_report" if state.get("ended_early") else "answer_evaluator"


def build_graph(checkpointer=None):
    """Compile the interview workflow. Pass a SqliteSaver/PostgresSaver for persistence across restarts."""
    g = StateGraph(InterviewState)
    for name in ("interview_configuration", "resume_analysis", "job_analysis", "interview_planner",
                 "interview_agent", "generate_question", "candidate_answer",
                 "answer_evaluator", "structured_evaluation", "feedback_generator", "interview_memory",
                 "adaptive_question_selection", "final_interview_report"):
        g.add_node(name, getattr(n, name))

    g.add_edge(START, "interview_configuration")
    g.add_edge("interview_configuration", "resume_analysis")
    g.add_edge("resume_analysis", "job_analysis")
    g.add_edge("job_analysis", "interview_planner")
    g.add_edge("interview_planner", "interview_agent")
    g.add_edge("interview_agent", "generate_question")
    g.add_edge("generate_question", "candidate_answer")
    g.add_conditional_edges("candidate_answer", route_after_answer,
                            ["answer_evaluator", "final_interview_report"])   # /quit shortcut
    g.add_edge("answer_evaluator", "structured_evaluation")
    g.add_edge("structured_evaluation", "feedback_generator")
    g.add_edge("feedback_generator", "interview_memory")
    g.add_conditional_edges("interview_memory", n.interview_complete)          # Interview Complete?
    g.add_edge("adaptive_question_selection", "generate_question")             # the loop
    g.add_edge("final_interview_report", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())
