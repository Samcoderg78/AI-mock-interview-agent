"""Cross-session candidate memory (JSON files). Within a session, memory lives in graph state."""
import json
import os
import re
from pathlib import Path


def home() -> Path:
    p = Path(os.getenv("MOCK_INTERVIEW_HOME", "~/.mock_interviewer")).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "candidate"


def _profile_path(name: str) -> Path:
    d = home() / "profiles"
    d.mkdir(exist_ok=True)
    return d / f"{_slug(name)}.json"


def load_profile(name: str) -> dict:
    p = _profile_path(name)
    data = json.loads(p.read_text()) if p.exists() else {"sessions": 0, "topics": {}}
    ranked = sorted(data["topics"].items(), key=lambda kv: kv[1]["avg"])
    data["weak"] = [t for t, v in ranked if v["avg"] < 6][:3]
    data["strong"] = [t for t, v in reversed(ranked) if v["avg"] >= 8][:3]
    return data


def update_profile(name: str, topic_scores: dict[str, list[float]]) -> dict:
    data = load_profile(name)
    data.pop("weak", None); data.pop("strong", None)
    data["sessions"] += 1
    for topic, scores in topic_scores.items():
        cur = data["topics"].get(topic, {"avg": 0.0, "n": 0})
        n = cur["n"] + len(scores)
        data["topics"][topic] = {"avg": round((cur["avg"] * cur["n"] + sum(scores)) / n, 2), "n": n}
    _profile_path(name).write_text(json.dumps(data, indent=2))
    return data


def save_report(name: str, markdown: str, stamp: str) -> str:
    d = home() / "reports"
    d.mkdir(exist_ok=True)
    path = d / f"{_slug(name)}-{stamp}.md"
    path.write_text(markdown)
    return str(path)
