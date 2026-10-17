"""Text embeddings. EMBED_PROVIDER=hf uses open-weight EmbeddingGemma; the default uses the Gemini API."""
import os
from typing import List

import httpx
import numpy as np

from .llm import API, TIMEOUT, LLMError, api_key

GEMINI_EMBED_MODEL = "gemini-embedding-001"
HF_EMBED_URL = (
    "https://router.huggingface.co/hf-inference/models/google/embeddinggemma-300m/pipeline/feature-extraction"
)
DIMS = 768
BATCH = 100


def _gemini(texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []
    for i in range(0, len(texts), BATCH):
        requests = [
            {
                "model": f"models/{GEMINI_EMBED_MODEL}",
                "content": {"parts": [{"text": t}]},
                "outputDimensionality": DIMS,
            }
            for t in texts[i : i + BATCH]
        ]
        resp = httpx.post(
            f"{API}/models/{GEMINI_EMBED_MODEL}:batchEmbedContents",
            params={"key": api_key()},
            json={"requests": requests},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            raise LLMError(f"embedding failed ({resp.status_code}): {resp.text[:300]}")
        out.extend(e["values"] for e in resp.json()["embeddings"])
    return out


def _hf(texts: List[str]) -> List[List[float]]:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise LLMError("HF_TOKEN is not set")
    resp = httpx.post(
        HF_EMBED_URL,
        headers={"Authorization": f"Bearer {token}"},
        json={"inputs": texts},
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        raise LLMError(f"EmbeddingGemma call failed ({resp.status_code}): {resp.text[:300]}")
    return resp.json()


def provider() -> str:
    return os.environ.get("EMBED_PROVIDER", "gemini").strip().lower()


def embed(texts: List[str]) -> List[List[float]]:
    """Return unit-length vectors (rounded to keep browser storage small)."""
    if not texts:
        return []
    raw = _hf(texts) if provider() == "hf" else _gemini(texts)
    mat = np.asarray(raw, dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    return np.round(mat, 4).tolist()


def cosine_matrix(a: List[List[float]], b: List[List[float]]) -> np.ndarray:
    """Vectors from embed() are already unit length, so cosine is a dot product."""
    return np.asarray(a, dtype=np.float32) @ np.asarray(b, dtype=np.float32).T
