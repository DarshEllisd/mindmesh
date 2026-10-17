"""Graph models and the validate-then-cap step every mode runs on model output."""
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Node(BaseModel):
    id: str
    mode: str
    kind: str
    title: str
    body: str = ""
    tags: List[str] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)
    vec: Optional[List[float]] = None


class Edge(BaseModel):
    source: str
    target: str
    label: str = ""


class Graph(BaseModel):
    nodes: List[Node]
    edges: List[Edge]


def _text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def finalize(
    raw: Any,
    mode: str,
    kinds: List[str],
    default_kind: str,
    max_nodes: int = 20,
    max_edges: int = 40,
) -> Graph:
    """Validate model JSON: fix unknown kinds, give nodes globally unique ids, drop edges
    that point at ids the model invented."""
    if not isinstance(raw, dict):
        raise ValueError("model output was not a JSON object")
    batch = uuid.uuid4().hex[:6]
    id_map: Dict[str, str] = {}
    nodes: List[Node] = []
    for i, item in enumerate(raw.get("nodes") or []):
        if not isinstance(item, dict) or not _text(item.get("title"), 1):
            continue
        if len(nodes) >= max_nodes:
            break
        local = str(item.get("id") or f"n{i}")
        global_id = f"{mode[0]}_{batch}_{len(nodes)}"
        id_map[local] = global_id
        kind = str(item.get("kind") or "").lower()
        tags = item.get("tags") if isinstance(item.get("tags"), list) else []
        nodes.append(
            Node(
                id=global_id,
                mode=mode,
                kind=kind if kind in kinds else default_kind,
                title=_text(item.get("title"), 90),
                body=_text(item.get("body"), 600),
                tags=[_text(t, 24).lower() for t in tags[:5] if t],
                extra={k: item[k] for k in ("order", "path", "url") if item.get(k) is not None},
            )
        )

    edges: List[Edge] = []
    seen = set()
    for item in raw.get("edges") or []:
        if not isinstance(item, dict) or len(edges) >= max_edges:
            continue
        src = id_map.get(str(item.get("source")))
        tgt = id_map.get(str(item.get("target")))
        if not src or not tgt or src == tgt or (src, tgt) in seen:
            continue
        seen.add((src, tgt))
        edges.append(Edge(source=src, target=tgt, label=_text(item.get("label"), 30)))
    return Graph(nodes=nodes, edges=edges)


def node_text(node: Node) -> str:
    return f"{node.title}. {node.body}"
