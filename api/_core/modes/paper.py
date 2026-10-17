"""Paper mode: a research PDF or figure screenshots -> claims, methods, results and citations."""
import base64
import io
import re
from typing import List, Optional

from pypdf import PdfReader

from ..chunk import condense
from ..llm import generate_json
from ..schema import Graph, finalize

KINDS = ["claim", "method", "result", "dataset", "concept", "citation"]
MAX_PAGES = 40

PROMPT = """You map a research paper into a graph a student can explore.
- kind: one of {kinds}
  claim = a contribution or thesis; method = technique or architecture; result = a measured
  finding (keep the numbers); dataset = data or benchmark; concept = background idea needed to
  follow the paper; citation = an important prior work it builds on or compares against.
- title: 3-8 words
- body: 1-3 plain-language sentences
- tags: 1-3 lowercase tags
Edges: "supports", "uses", "evaluated on", "builds on", "compares to", "defines".
The first node should be the paper itself (kind "claim", title = paper title, body = one-line summary).
Return 6 to 18 nodes. Figures, if given as images, are from the same paper.

Reply with ONLY this JSON:
{{"nodes": [{{"id": "n1", "kind": "claim", "title": "...", "body": "...", "tags": ["..."]}}],
  "edges": [{{"source": "n2", "target": "n1", "label": "supports"}}]}}

PAPER CONTENT:
{text}"""


def pdf_text(pdf_b64: str) -> str:
    data = base64.b64decode(re.sub(r"^data:[\w/+.-]+;base64,", "", pdf_b64))
    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages[:MAX_PAGES]]
    text = "\n\n".join(pages)
    # Everything after the reference list is mostly noise for this purpose, but keep the
    # reference list itself short so citations can still be named.
    refs = re.search(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)
    if refs:
        text = text[: refs.start()] + "\n\nREFERENCES (excerpt):\n" + text[refs.end() : refs.end() + 3000]
    return text


def run(pdf_b64: Optional[str], text: str, images: List[str]) -> Graph:
    content = text
    if pdf_b64:
        content = pdf_text(pdf_b64) + "\n\n" + text
    if not content.strip() and not images:
        raise ValueError("upload a PDF, paste text, or add figure images")
    content = condense(content, "the paper's claims, methods, results, datasets and key citations")
    raw = generate_json(PROMPT.format(kinds=", ".join(KINDS), text=content or "(see images)"), images=images)
    return finalize(raw, "paper", KINDS, "concept", max_nodes=18)
