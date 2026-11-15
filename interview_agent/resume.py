"""Resume text extraction (PDF or plain text). Text-based PDFs only; scanned images need OCR."""
import io
import re
from pathlib import Path

MAX_CHARS = 12000     # keeps prompts small and cheap


def extract_text(data: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join((page.extract_text() or "") for page in PdfReader(io.BytesIO(data)).pages)
    else:
        text = data.decode("utf-8", errors="ignore")
    return re.sub(r"\n{3,}", "\n\n", text).strip()[:MAX_CHARS]


def load_resume(path: str) -> str:
    p = Path(path).expanduser()
    return extract_text(p.read_bytes(), p.name)
