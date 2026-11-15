"""Streamlit web UI:  streamlit run app.py"""
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from interview_agent.graph import build_graph
from interview_agent.resume import extract_text

load_dotenv()
st.set_page_config(page_title="AI Mock Interview Agent", page_icon="🎤", layout="centered")
ss = st.session_state


def cfg():
    return {"configurable": {"thread_id": ss.thread}}


def advance(payload) -> None:
    """Run the graph until the next interrupt (a question waiting for an answer) or the end."""
    with st.spinner("Interviewer is thinking..."):
        res = ss.graph.invoke(payload, cfg())
    pending = res.get("__interrupt__")
    ss.pending = pending[0].value if pending else None
    ss.report = None if pending else res.get("report")


def read_upload(label: str, key: str) -> str:
    up = st.file_uploader(label, type=["pdf", "txt", "md"], key=key)
    if up is None:
        return ""
    text = extract_text(up.getvalue(), up.name)
    if text:
        st.success(f"Loaded ({len(text)} characters).")
    else:
        st.warning("Could not read text from this file (scanned PDFs are not supported). Try a text-based PDF.")
    return text


with st.sidebar:
    st.header("Interview setup")
    st.caption("Give a job description, a resume, or both. Questions are built from them, for any profession.")
    name = st.text_input("Your name", "Candidate")
    role = st.text_input("Role (optional)", "", placeholder="Taken from the job description if empty")
    level = st.selectbox("Level", ["auto", "junior", "mid", "senior"], help="auto = inferred from the job / resume")
    st.subheader("Job description")
    jd_pasted = st.text_area("Paste the job posting", height=120, label_visibility="collapsed",
                             placeholder="Paste the job description here...")
    jd_file = read_upload("...or upload it (PDF/text)", "jd")
    st.subheader("Resume")
    resume_text = read_upload("Upload your resume (PDF/text)", "resume")
    focus_raw = st.text_area("Topics you want to practice (optional, one per line)", height=70)
    jd_text = (jd_pasted.strip() or jd_file)[:12000]
    focus = [x.strip() for x in focus_raw.splitlines() if x.strip()]
    ready = bool(role.strip() or jd_text or resume_text or focus)
    st.caption("The interview ends on its own once everything planned is covered. You can end it any time.")
    if st.button("Start new interview", type="primary", disabled=not ready):
        ss.graph, ss.thread, ss.pending, ss.report = build_graph(), str(uuid.uuid4()), None, None
        advance({"config": {"candidate_name": name, "role": role, "level": level, "job_description": jd_text,
                            "resume_text": resume_text, "focus_topics": focus}})
        st.rerun()
    if not ready:
        st.info("Add a role, job description, resume or focus topic to begin.")

st.title("🎤 AI Mock Interview Agent")
if "graph" not in ss:
    st.info("Set up your interview in the sidebar and press **Start new interview**.")
    st.stop()

state = ss.graph.get_state(cfg()).values
plan = state.get("interview_plan")
if plan:
    with st.expander(f"Interview plan: {plan['level']}-level {plan['role']}"):
        for t in plan["topics"]:
            st.markdown(f"- **{t['name']}** ({t['style']})" + (f": {t['rationale']}" if t.get("rationale") else ""))

for t in state.get("turns", []):
    q, ev, fb = t["question"], t["evaluation"], t["feedback"]
    with st.chat_message("assistant"):
        st.markdown(f"**{q['text']}**")
        st.caption(f"{q['topic']} | {q['style']} | {q['difficulty']} | {q['kind']}")
    with st.chat_message("user"):
        st.write(t["answer"] or "_(skipped)_")
    with st.chat_message("assistant"):
        st.markdown(f"**Score {ev['overall']}/10**  \n{fb['summary']}  \n💡 {fb['improvement_tip']}")
        with st.expander("Model answer"):
            st.write(fb["model_answer"])

if ss.pending:
    p = ss.pending
    with st.chat_message("assistant"):
        more = f"about {p['remaining']} more planned" if p["remaining"] else "last planned question"
        st.markdown(f"**Question {p['number']}** ({more})  \n{p['question']}")
        st.caption(f"{p['topic']} | {p['style']} | {p['difficulty']} | {p['kind']}")
    c1, c2 = st.columns(2)
    if c1.button("Skip question"):
        advance(Command(resume="/skip")); st.rerun()
    if c2.button("End interview now"):
        advance(Command(resume="/quit")); st.rerun()
    if answer := st.chat_input("Type your answer..."):
        advance(Command(resume=answer)); st.rerun()
elif ss.report:
    st.success("Interview complete!")
    st.markdown(ss.report)
    st.download_button("Download report (.md)", ss.report, file_name="interview_report.md")
