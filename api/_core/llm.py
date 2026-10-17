"""Gemma 4 calls over the Gemini REST API, plus JSON cleanup for model output."""
import json
import logging
import os
import re
import time
from typing import Any, List, Optional

import httpx

API = "https://generativelanguage.googleapis.com/v1beta"
TIMEOUT = httpx.Timeout(55.0, connect=10.0)

# httpx logs every request URL at INFO level. Keep that out of the function logs.
logging.getLogger("httpx").setLevel(logging.WARNING)


class LLMError(RuntimeError):
    pass


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise LLMError("GEMINI_API_KEY is not set")
    return key


def auth_headers() -> dict:
    """Send the key as a header, never in the URL, so it cannot end up in request logs."""
    return {"x-goog-api-key": api_key()}


FALLBACK_MODELS = ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]
RETRYABLE = {429, 500, 502, 503, 504}
BUDGET_SECONDS = float(os.environ.get("GEMMA_BUDGET_SECONDS", "45"))  # stays under Vercel's 60s
ATTEMPT_SECONDS = 20.0  # healthy calls take 3-15s; a stalled one is abandoned and retried


def model_id() -> str:
    return os.environ.get("GEMMA_MODEL", FALLBACK_MODELS[0]).strip()


def thinking_level() -> str:
    # "minimal" keeps extraction fast; set GEMMA_THINKING=high for harder reasoning.
    return os.environ.get("GEMMA_THINKING", "minimal").strip()


def _image_part(data_url: str) -> dict:
    """Turn a data URL (data:image/jpeg;base64,...) into an inline_data part."""
    match = re.match(r"data:([\w/+.-]+);base64,(.+)", data_url, re.S)
    if not match:
        raise LLMError("images must be base64 data URLs")
    return {"inline_data": {"mime_type": match.group(1), "data": match.group(2)}}


def generate(
    prompt: str,
    images: Optional[List[str]] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> str:
    # Gemma models on the Gemini API take no system instruction, so everything goes in the user turn.
    parts: List[dict] = [_image_part(img) for img in (images or [])]
    parts.append({"text": prompt})
    config = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if thinking_level():
        config["thinkingConfig"] = {"thinkingLevel": thinking_level()}
    body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}

    # The hosted Gemma endpoints fail intermittently (fast 500s, 429 rate limits, stalled
    # requests), so retry within a time budget. Every third attempt tries the other Gemma 4 size.
    primary = model_id()
    fallback = next((m for m in FALLBACK_MODELS if m != primary), primary)
    deadline = time.monotonic() + BUDGET_SECONDS
    attempt = 0
    last_error = "no attempt made"
    while time.monotonic() < deadline - 2:
        model = fallback if attempt % 3 == 2 else primary
        attempt += 1
        remaining = deadline - time.monotonic()
        try:
            resp = httpx.post(
                f"{API}/models/{model}:generateContent",
                headers=auth_headers(),
                json=body,
                timeout=httpx.Timeout(min(ATTEMPT_SECONDS, remaining), connect=10.0),
            )
        except httpx.TimeoutException:
            last_error = f"{model} timed out"
            continue
        if resp.status_code == 200:
            break
        last_error = f"{model} returned {resp.status_code}: {resp.text[:200]}"
        if resp.status_code not in RETRYABLE:
            raise LLMError(f"Gemma call failed: {last_error}")
        if resp.status_code == 429:
            # Rate limited: wait as asked (or a few seconds) instead of hammering the quota.
            try:
                wait = float(resp.headers.get("retry-after", ""))
            except ValueError:
                wait = 3.0 * attempt
            time.sleep(max(0.0, min(wait, 8.0, deadline - time.monotonic() - 3)))
        else:
            time.sleep(min(0.5 * attempt, 2))
    else:
        busy = "rate limit reached" if "429" in last_error else last_error
        raise LLMError(f"Gemma is busy right now ({busy}). Try again in a moment.")
    data = resp.json()
    try:
        parts_out = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError):
        raise LLMError(f"Gemma returned no content: {json.dumps(data)[:300]}")
    # Skip any "thought" parts some models emit and keep the answer text.
    return "".join(p.get("text", "") for p in parts_out if not p.get("thought"))


def clean_json(text: str) -> Any:
    """Strip markdown fences and surrounding chatter, then parse the first JSON value."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in model output")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Common slip: trailing commas before a closing bracket.
        return json.loads(re.sub(r",\s*([}\]])", r"\1", candidate))


def generate_json(prompt: str, images: Optional[List[str]] = None, **kwargs) -> Any:
    """Ask for JSON and retry once with a stricter reminder if parsing fails."""
    out = generate(prompt, images, **kwargs)
    try:
        return clean_json(out)
    except ValueError:
        retry = prompt + "\n\nIMPORTANT: Reply with ONLY valid JSON. No prose, no markdown fences."
        return clean_json(generate(retry, images, **kwargs))


def list_models() -> List[str]:
    resp = httpx.get(f"{API}/models", headers=auth_headers(), params={"pageSize": 1000}, timeout=TIMEOUT)
    resp.raise_for_status()
    return [m["name"].split("/", 1)[-1] for m in resp.json().get("models", [])]
