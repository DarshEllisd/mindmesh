"""Chunk long inputs and condense them, instead of truncating to the first N words."""
from concurrent.futures import ThreadPoolExecutor
from typing import List

from .llm import generate

CHUNK_CHARS = 15000
OVERLAP = 300
DIRECT_LIMIT = 60000  # about 15k tokens; a typical paper goes to the model in one call
MAX_CHUNKS = 8  # all summarized in one parallel round, inside the serverless time limit


def chunk(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP) -> List[str]:
    text = text.strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        # Prefer to break on a paragraph or sentence boundary.
        if end < len(text):
            cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def condense(text: str, focus: str) -> str:
    """Return text short enough for one prompt. Long text is split into chunks, each chunk is
    summarized in parallel (map), and the summaries are joined in order (reduce)."""
    if len(text) <= DIRECT_LIMIT:
        return text
    parts = chunk(text)
    if len(parts) > MAX_CHUNKS:
        # Sample evenly across the document so the end is not lost.
        step = len(parts) / MAX_CHUNKS
        parts = [parts[int(i * step)] for i in range(MAX_CHUNKS)]

    def summarize(indexed):
        i, part = indexed
        prompt = (
            f"Section {i + 1} of {len(parts)} of a longer document.\n"
            f"Extract the key points relevant to: {focus}.\n"
            "Write 5-10 dense bullet points. Keep names, numbers and technical terms exactly.\n\n"
            f"---\n{part}\n---"
        )
        return f"[Section {i + 1}]\n" + generate(prompt, max_tokens=800)

    with ThreadPoolExecutor(max_workers=MAX_CHUNKS) as pool:
        return "\n\n".join(pool.map(summarize, enumerate(parts)))
