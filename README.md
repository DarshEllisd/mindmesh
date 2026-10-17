# Mindmesh

**Notes, code repositories and research papers in one linked knowledge mesh, built on Gemma 4.**

Most note tools keep your lecture notes, the repo you're studying and the paper behind it in separate places. Mindmesh breaks each source into small, single-idea nodes. It embeds every node and links related ideas, **including across sources**. For example, your handwritten note on "self-attention" links to the paper's *Scaled Dot-Product Attention* method and to the `model.py` module that implements it. You can then ask questions of the whole mesh and get answers that cite the nodes they used.

Built at Hacktoberfest Hack Day Zuarinagar x SEDS Celestia (17 Oct 2026).

## Modes

| Mode | Input | What Gemma 4 produces |
|---|---|---|
| **Notes** | Typed text and/or photos of handwritten notes or whiteboards (multimodal) | Single-idea notes (idea, fact, definition, task, question) and the relationships between them |
| **Repo** | A public GitHub URL | A module map, a numbered **start-here reading order** for newcomers, and the repo's *good first issues*, each matched by embedding similarity to the module it most likely touches |
| **Paper** | A PDF, pasted text, or figure screenshots | Claims, methods, results, datasets, background concepts and key citations, with "supports / uses / evaluated on" links |
| **Edit the map** | Ctrl/⌘-click or Shift-drag to select several ideas | Remove them in one go (button or Delete key), with **Undo** for 8 seconds (or Ctrl+Z) |
| **Ask** | A question | An answer drawn only from the nearest nodes, with clickable citations that are highlighted in the graph |

## How it works

```
Browser (static UI, vis-network graph, mesh kept in localStorage)
   │ 1. input (text / images / repo URL / PDF)
   ▼
FastAPI on Vercel (stateless)
   ├─ long inputs → chunk → parallel summaries per chunk (map-reduce, not truncation)
   ├─ Gemma 4 (multimodal) → JSON nodes + edges → validate and cap
   │     (unknown kinds fixed, ids made unique, edges to invented ids dropped)
   └─ embed every node (768-d, unit length)
   │ 2. nodes + vectors
   ▼
Browser: cosine similarity between new nodes and the existing mesh → top-3 neighbours per node
   │ 3. only those candidate pairs (≤30) → /api/relate → Gemma 4 labels or rejects each pair
   ▼
Labelled cross-source links, drawn as red dashed edges
```

Design choices:
- **Incremental linking.** Each new node is compared only with its nearest neighbours. Mindmesh never sends the whole board to the model, so the cost of an update stays the same as the mesh grows.
- **A model check after vector search.** Similar embeddings only propose a link. Gemma decides whether it's a real relationship and names it.
- **Embeddings first, Gemma when it adds value.** Ask runs a vector search in the browser and shows the closest ideas straight away, using one cheap embedding call. Gemma writes an answer only when you click **Explain with Gemma**. Reworded questions (cosine ≥ 0.88; in testing, paraphrases scored about 0.92 and unrelated questions about 0.58) reuse the earlier answer with no model call. When linking, pairs at similarity 0.8 or above are linked directly as "same idea"; only the uncertain pairs go to Gemma, and the call is skipped when there are none.
- **No database.** The server keeps no state, and the mesh lives in your browser. Use **Export** to download it as JSON.
- **Fast and resilient Gemma calls.** Thinking is set to `minimal` (`GEMMA_THINKING`). With default thinking, extraction regularly took over 60s; with minimal it takes about 5–15s. Hosted Gemma endpoints sometimes return quick 500 errors, so each call retries within a 45s budget and alternates between the two Gemma 4 sizes.

Measured locally: notes about 12s, whiteboard photo about 34s, the nanoGPT repo about 37s, the 15-page *Attention Is All You Need* PDF about 31s, relation labelling about 4s, Ask about 3s.

## AI models and licenses

| Purpose | Model | License / terms |
|---|---|---|
| Extraction, image understanding, relation labelling, Q&A | **Gemma 4** (open-weight, Google DeepMind), called through the Gemini API. The default is `gemma-4-26b-a4b-it` (mixture-of-experts, fast), with automatic fallback to `gemma-4-31b-it`. Set with `GEMMA_MODEL`. | [Gemma terms / license](https://ai.google.dev/gemma/terms), [model overview](https://ai.google.dev/gemma/docs) |
| Embeddings (open option) | **EmbeddingGemma 300M** (open-weight) through Hugging Face Inference, with `EMBED_PROVIDER=hf` | [Model card and license](https://huggingface.co/google/embeddinggemma-300m) |
| Embeddings (default) | `gemini-embedding-001` through the Gemini API | [Gemini API terms](https://ai.google.dev/gemini-api/terms) |

Gemma 4 does all the generation, so the system's reasoning runs on an open-weight model. You can switch embeddings to the open-weight EmbeddingGemma with a single environment variable.

## Run it locally

Requirements: Python 3.9 or newer, and a [Google AI Studio API key](https://aistudio.google.com/apikey).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt uvicorn python-dotenv
cp .env.example .env        # then set GEMINI_API_KEY
uvicorn api.index:app --reload --port 8000
```

Open http://localhost:8000. The same server serves the UI locally. To see which Gemma models your key can use, open http://localhost:8000/api/models.

### Deploy to Vercel

1. Import the repository in Vercel. Vercel detects the FastAPI app in `api/index.py` and routes requests to it, while `public/` is served as static files. `vercel.json` only raises the function time limit to 60s. Don't add a rewrite to `/api/index`: FastAPI would then receive the rewritten path and return 404 for every route.
2. Set `GEMINI_API_KEY`, and optionally `GEMMA_MODEL`, `EMBED_PROVIDER`, `HF_TOKEN` and `GITHUB_TOKEN`, under **Project → Settings → Environment Variables**.
3. Deploy.

4. Optional: open **Project → Analytics → Enable** to turn on Vercel Web Analytics. The page already includes the cookieless analytics snippet, which loads only on deployed hosts, not on localhost. Page views are counted on every plan. A few custom events are also sent (`Map source`, `Search map`, `Explain with Gemma`, `Reuse answer`, `Remove ideas`), carrying only the source type and counts, never your notes, questions or URLs. Custom events are recorded on Vercel Pro.

Limits: request bodies are capped at 4.5MB, so images are downscaled in the browser and PDFs should be under about 3MB. Each request has 60s.

## Key dependencies

- [FastAPI](https://fastapi.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/): API and validation
- [httpx](https://www.python-httpx.org/): Gemini, Hugging Face and GitHub REST calls
- [NumPy](https://numpy.org/): vector maths
- [pypdf](https://pypdf.readthedocs.io/): PDF text extraction
- [vis-network](https://visjs.github.io/vis-network/docs/network/): graph rendering in the browser
- [GSAP](https://gsap.com/): interface motion (the intro, ideas plotting in, route pulses, panel transitions). Motion is turned off under `prefers-reduced-motion`, and the app works without it.
- GitHub REST API: repository metadata, tree, README and issues

## Project layout

```
api/index.py            FastAPI routes (single Vercel function)
api/_core/llm.py        Gemma 4 calls and JSON cleanup with a retry
api/_core/embed.py      embeddings (Gemini or EmbeddingGemma) and cosine similarity
api/_core/chunk.py      chunking and parallel summaries for long inputs
api/_core/schema.py     Node/Edge models and the validate-then-cap step
api/_core/modes/        notes.py, repo.py, paper.py: one prompt and one validator per mode
public/                 index.html, app.js (graph, linking, Ask), motion.js (GSAP), style.css
```

## License

[MIT](LICENSE)
