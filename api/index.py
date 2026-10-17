"""Mindmesh API. Runs as a single Vercel Python function; every /api/* path is rewritten here."""
import os
import re
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

try:  # local development reads .env; on Vercel the variables come from project settings
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except ImportError:
    pass

from _core import embed as embedding  # noqa: E402
from _core.llm import LLMError, generate, generate_json, list_models, model_id  # noqa: E402
from _core.modes import notes, paper, repo  # noqa: E402
from _core.schema import Graph, node_text  # noqa: E402

app = FastAPI(title="Mindmesh")

MAX_IMAGES = 4
MAX_PAIRS = 30
MAX_CONTEXT = 12


@app.exception_handler(LLMError)
async def llm_error(_, exc: LLMError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.exception_handler(ValueError)
async def bad_input(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def with_vectors(graph: Graph) -> Graph:
    """Embed every node that does not have a vector yet, so the browser can link it."""
    missing = [n for n in graph.nodes if n.vec is None]
    for node, vec in zip(missing, embedding.embed([node_text(n) for n in missing])):
        node.vec = vec
    return graph


class NotesIn(BaseModel):
    text: str = ""
    images: List[str] = Field(default_factory=list)


class RepoIn(BaseModel):
    url: str


class PaperIn(BaseModel):
    pdf: Optional[str] = None
    text: str = ""
    images: List[str] = Field(default_factory=list)


class EmbedIn(BaseModel):
    texts: List[str]


class Brief(BaseModel):
    id: str
    title: str
    body: str = ""
    mode: str = ""


class Pair(BaseModel):
    a: Brief
    b: Brief


class RelateIn(BaseModel):
    pairs: List[Pair]


class AskIn(BaseModel):
    question: str
    context: List[Brief]


@app.get("/api/health")
def health():
    return {"ok": True, "model": model_id(), "embeddings": embedding.provider()}


@app.get("/api/models")
def models():
    return {"gemma": [m for m in list_models() if "gemma" in m]}


@app.post("/api/notes")
def notes_route(body: NotesIn):
    if not body.text.strip() and not body.images:
        raise HTTPException(400, "add some text or at least one image")
    return with_vectors(notes.run(body.text, body.images[:MAX_IMAGES]))


@app.post("/api/repo")
def repo_route(body: RepoIn):
    return with_vectors(repo.run(body.url))


@app.post("/api/paper")
def paper_route(body: PaperIn):
    return with_vectors(paper.run(body.pdf, body.text, body.images[:MAX_IMAGES]))


@app.post("/api/embed")
def embed_route(body: EmbedIn):
    return {"vectors": embedding.embed([t[:2000] for t in body.texts[:100]])}


RELATE_PROMPT = """For each numbered pair of notes below, decide whether they are meaningfully
related. They may come from different sources (personal notes, a code repository, a research paper).
If related, give a short relation label of 1-3 words describing how A relates to B
(e.g. "explains", "applies", "example of", "same topic", "contradicts", "implements").
If they are only superficially similar, use an empty label.

Reply with ONLY a JSON array, one entry per pair, in order:
[{{"pair": 1, "label": "explains"}}, {{"pair": 2, "label": ""}}]

PAIRS:
{pairs}"""


@app.post("/api/relate")
def relate_route(body: RelateIn):
    """Label edges for each new node's top-k nearest neighbours only, instead of sending the
    whole board to the model on every update."""
    pairs = body.pairs[:MAX_PAIRS]
    if not pairs:
        return {"edges": []}
    listing = "\n\n".join(
        f"{i + 1}. A [{p.a.mode}] {p.a.title}: {p.a.body[:200]}\n   B [{p.b.mode}] {p.b.title}: {p.b.body[:200]}"
        for i, p in enumerate(pairs)
    )
    raw = generate_json(RELATE_PROMPT.format(pairs=listing), temperature=0.1, max_tokens=1500)
    edges = []
    for item in raw if isinstance(raw, list) else []:
        try:
            pair = pairs[int(item.get("pair")) - 1]
        except (TypeError, ValueError, IndexError, AttributeError):
            continue
        label = str(item.get("label") or "").strip()[:30]
        if label:
            edges.append({"source": pair.a.id, "target": pair.b.id, "label": label})
    return {"edges": edges}


ASK_PROMPT = """Answer the question using ONLY the notes below. Cite the notes you use inline with
their id in square brackets, like [n_abc123_0]. Give a direct answer first. Reasonable inferences
from the notes are fine (for example, the file that holds the model architecture is where its
layers are implemented); state them plainly. Only if the notes have nothing relevant, say so
briefly and suggest what to add. Keep the answer under 150 words.

NOTES:
{notes}

QUESTION: {question}"""


@app.post("/api/ask")
def ask_route(body: AskIn):
    if not body.question.strip():
        raise HTTPException(400, "ask a question")
    context = body.context[:MAX_CONTEXT]
    listing = "\n".join(f"[{c.id}] ({c.mode}) {c.title}: {c.body[:500]}" for c in context)
    answer = generate(ASK_PROMPT.format(notes=listing, question=body.question.strip()), max_tokens=700)
    known = {c.id for c in context}
    cited = [cid for cid in dict.fromkeys(re.findall(r"\[([\w-]+)\]", answer)) if cid in known]
    return {"answer": answer.strip(), "cited": cited}


# Local development: serve the static UI from the same server. On Vercel, public/ is served by the CDN.
_public = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public")
if os.environ.get("VERCEL") is None and os.path.isdir(_public):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_public, html=True), name="public")
