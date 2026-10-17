"""Notes mode: text dumps and photos of whiteboards or handwritten notes -> atomic notes."""
from typing import List

from ..chunk import condense
from ..llm import generate_json
from ..schema import Graph, finalize

KINDS = ["idea", "fact", "definition", "task", "question"]

PROMPT = """You turn raw notes into an atomic knowledge graph.
Input may be typed text, photos of handwritten notes, or photos of a whiteboard (read every
image carefully, including diagrams and arrows).

Split the content into small, self-contained notes: one idea per note.
- kind: one of {kinds}
- title: 3-8 words
- body: 1-3 sentences that make sense on their own (expand shorthand, fix OCR slips)
- tags: 1-3 lowercase topic tags
Add edges only for relationships stated or clearly implied in the input
(labels like "part of", "causes", "example of", "depends on", "contrasts").
Return 3 to 15 notes.

Reply with ONLY this JSON:
{{"nodes": [{{"id": "n1", "kind": "idea", "title": "...", "body": "...", "tags": ["..."]}}],
  "edges": [{{"source": "n1", "target": "n2", "label": "..."}}]}}

INPUT TEXT:
{text}"""


def run(text: str, images: List[str]) -> Graph:
    text = condense(text, "the main ideas, facts and tasks") if text.strip() else "(see images)"
    raw = generate_json(PROMPT.format(kinds=", ".join(KINDS), text=text), images=images)
    return finalize(raw, "notes", KINDS, "idea", max_nodes=15)
