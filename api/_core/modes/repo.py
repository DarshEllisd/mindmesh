"""Repo mode: a GitHub repository -> module map, a "start here" path, and good first issues
matched to the module they most likely touch."""
import base64
import os
import re
import uuid
from typing import Dict, List, Optional, Tuple

import httpx

from ..chunk import condense
from ..embed import cosine_matrix, embed
from ..llm import generate_json
from ..schema import Edge, Graph, Node, finalize, node_text

KINDS = ["module", "entrypoint", "concept", "config", "docs"]
GH = "https://api.github.com"
MANIFESTS = ["package.json", "pyproject.toml", "requirements.txt", "go.mod", "Cargo.toml", "setup.py", "pom.xml"]
SKIP_DIRS = re.compile(r"(^|/)(node_modules|vendor|dist|build|\.git|__pycache__|\.next|target)(/|$)")
ISSUE_LABELS = ["good first issue", "good-first-issue", "help wanted"]
MAX_TREE = 350
MATCH_MIN = 0.3

PROMPT = """You help a first-time contributor understand an open-source repository.
Using the metadata, README, manifests and file tree below, map the codebase.
- kind: one of {kinds}
- title: the module or concept name (short)
- body: 1-2 sentences on what it does and why a contributor would care
- path: the main directory or file for this node, exactly as in the tree (omit for concepts)
- order: for 3-6 nodes, a number 1..N giving the best reading order for a newcomer ("start here" path); omit for others
- tags: 1-3 lowercase tags
Edges: "imports", "calls", "configures", "documents", "part of".
Return 6 to 16 nodes. Only name paths that exist in the tree.

Reply with ONLY this JSON:
{{"nodes": [{{"id": "n1", "kind": "entrypoint", "title": "...", "body": "...", "path": "src/main.py", "order": 1, "tags": ["..."]}}],
  "edges": [{{"source": "n1", "target": "n2", "label": "imports"}}]}}

REPOSITORY: {full_name}
DESCRIPTION: {description}
LANGUAGE: {language}

README:
{readme}

MANIFESTS:
{manifests}

FILE TREE:
{tree}"""


def parse_repo(url: str) -> Tuple[str, str]:
    match = re.search(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/|$|\?|#)", url.strip() + "/")
    if not match:
        match = re.fullmatch(r"([\w.-]+)/([\w.-]+)", url.strip())
    if not match:
        raise ValueError("expected a GitHub URL like https://github.com/owner/repo")
    return match.group(1), match.group(2)


def _client() -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "mindmesh"}
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url=GH, headers=headers, timeout=20.0)


def _file(gh: httpx.Client, owner: str, repo: str, path: str) -> Optional[str]:
    resp = gh.get(f"/repos/{owner}/{repo}/contents/{path}")
    if resp.status_code != 200:
        return None
    return base64.b64decode(resp.json().get("content", "")).decode("utf-8", "replace")


def fetch(owner: str, repo: str) -> Dict:
    with _client() as gh:
        meta = gh.get(f"/repos/{owner}/{repo}")
        if meta.status_code == 404:
            raise ValueError("repository not found (it must be public)")
        if meta.status_code == 403:
            raise ValueError("GitHub rate limit hit; set GITHUB_TOKEN")
        meta.raise_for_status()
        info = meta.json()

        readme = gh.get(f"/repos/{owner}/{repo}/readme", headers={"Accept": "application/vnd.github.raw"})
        tree_resp = gh.get(f"/repos/{owner}/{repo}/git/trees/{info['default_branch']}", params={"recursive": "1"})
        paths = [
            item["path"]
            for item in (tree_resp.json().get("tree", []) if tree_resp.status_code == 200 else [])
            if not SKIP_DIRS.search(item["path"])
        ]
        top_level = {p for p in paths if "/" not in p}
        manifests = {}
        for name in MANIFESTS:
            if name in top_level:
                content = _file(gh, owner, repo, name)
                if content:
                    manifests[name] = content[:2000]

        issues = []
        for label in ISSUE_LABELS:
            resp = gh.get(f"/repos/{owner}/{repo}/issues", params={"labels": label, "state": "open", "per_page": 12})
            if resp.status_code == 200:
                issues = [i for i in resp.json() if "pull_request" not in i]
            if issues:
                break

    return {
        "info": info,
        "readme": readme.text if readme.status_code == 200 else "",
        "paths": paths,
        "manifests": manifests,
        "issues": issues,
    }


def _issue_nodes(issues: List[dict], prefix: str) -> List[Node]:
    return [
        Node(
            id=f"{prefix}_i{i}",
            mode="repo",
            kind="issue",
            title=f"#{issue['number']} {issue['title']}"[:90],
            body=re.sub(r"\s+", " ", issue.get("body") or "")[:400],
            tags=[lbl["name"].lower()[:24] for lbl in issue.get("labels", [])[:3]],
            extra={"url": issue["html_url"]},
        )
        for i, issue in enumerate(issues)
    ]


def run(url: str) -> Graph:
    owner, repo = parse_repo(url)
    data = fetch(owner, repo)
    info = data["info"]
    paths = data["paths"]
    # Keep the tree readable: directories first and shallow files, then trim.
    tree = sorted(paths, key=lambda p: (p.count("/"), p))[:MAX_TREE]
    prompt = PROMPT.format(
        kinds=", ".join(KINDS),
        full_name=info["full_name"],
        description=info.get("description") or "",
        language=info.get("language") or "",
        readme=condense(data["readme"], "what the project does, its architecture and how to contribute")[:60000],
        manifests="\n\n".join(f"--- {k}\n{v}" for k, v in data["manifests"].items()) or "(none)",
        tree="\n".join(tree),
    )
    graph = finalize(generate_json(prompt), "repo", KINDS, "module", max_nodes=16)

    # A root node ties the map together and carries the repo link and license.
    root = Node(
        id=f"r_{uuid.uuid4().hex[:6]}_root",
        mode="repo",
        kind="repository",
        title=info["full_name"],
        body=(info.get("description") or "")[:300],
        tags=[t[:24] for t in (info.get("topics") or [])[:3]],
        extra={
            "url": info["html_url"],
            "license": (info.get("license") or {}).get("spdx_id") or "none",
            "stars": info.get("stargazers_count"),
        },
    )
    for node in graph.nodes:
        if node.extra.get("path"):
            node.extra["url"] = f"{info['html_url']}/tree/{info['default_branch']}/{node.extra['path']}"
    # Connect nodes nothing else points to, so the map hangs off the repository root.
    linked = {e.source for e in graph.edges} | {e.target for e in graph.edges}
    for node in graph.nodes:
        if node.id not in linked or node.kind == "entrypoint":
            graph.edges.append(Edge(source=node.id, target=root.id, label="part of"))
    graph.nodes.insert(0, root)

    issues = _issue_nodes(data["issues"], root.id)
    modules = [n for n in graph.nodes if n.kind in ("module", "entrypoint", "config", "docs")]
    if issues and modules:
        vectors = embed([node_text(n) for n in modules + issues])
        for node, vec in zip(modules + issues, vectors):
            node.vec = vec
        sims = cosine_matrix(vectors[len(modules):], vectors[: len(modules)])
        for issue, row in zip(issues, sims):
            best = int(row.argmax())
            target = modules[best].id if row[best] >= MATCH_MIN else root.id
            graph.edges.append(Edge(source=issue.id, target=target, label="likely touches"))
        graph.nodes.extend(issues)
    return graph
