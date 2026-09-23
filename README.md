# Hotel guest assistant

A full-stack AI chat assistant for The Grand Azure Heritage Resort and Spa, Candolim, Goa. Guests ask about rooms, amenities, policies, and availability in plain language and get grounded answers with sources.

Live demo: https://hotel-guest-assistant-pi.vercel.app/
Backend: https://hotel-guest-assistant-xybo.onrender.com

## What it does

- Answers questions about the resort from a verified knowledge base (property, 6 amenities, 4 room types, 7 policies, 11 FAQs).
- Checks live room availability when given check-in date, check-out date, and guest count, and shows room cards with pricing.
- Asks for dates with an inline picker when they are missing.
- Streams replies token by token and shows source citations under each answer.
- Refuses to invent facts. Out-of-scope questions get a concierge contact instead of a guess.
- Blocks prompt injection attempts before they reach the model.

## Run it locally

You need Python 3.12 with `uv`, Node 18+, and a free Google AI Studio key (`GEMINI_API_KEY`).

Backend:

```bash
cd backend
uv sync
cp .env.example .env   # add your GEMINI_API_KEY
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

Frontend (second terminal):

```bash
cd frontend
npm install
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 npm run dev
```

Open http://localhost:3000.

Everything except the LLM call is free and local: FastEmbed embeddings, Qdrant, FlashRank reranking. The only paid-possible part is Gemini generation, which has a free tier. Without a key the backend runs an offline mock engine instead.

## How it works

A user message goes through this pipeline:

1. Prompt injection check, rejected early if malicious.
2. HyDE expands the short question into a hypothetical handbook passage for better matching.
3. Hybrid search runs dense Qdrant vectors (FastEmbed, local) and BM25 in parallel, fused with Reciprocal Rank Fusion into the top 10 chunks.
4. FlashRank (local cross-encoder) narrows the 10 down to the top 3.
5. Gemini writes the reply grounded only on those 3 chunks, or calls `check_room_availability` when dates are present.
6. Pricing and availability math stays in deterministic code, never in the model. Any LLM or retrieval failure falls back to the local engine rather than erroring.

The same flow as a diagram:

```
              Guest browser (Next.js on Vercel)
                          |
                          |  POST /api/chat/stream (SSE)
                          v
              FastAPI backend (Render)
                          |
        +-----------------+------------------+
        |                 |                  |
   1. Guard         2. Retrieve          3. Generate
injection check     top 3 chunks         reply + tools
        |                 |                  |
        |    HyDE expands the question       |
        |    via Gemini                      |  Gemini writes from the
        |                 |                  |  3 chunks only, or calls
        |    Hybrid search, top 10:          |  check_room_availability
        |    Qdrant vectors (FastEmbed,      |  when dates are present
        |    local) + BM25, fused            |
        |    with RRF                        |  Pricing math stays in
        |                 |                  |  code, never the model
        |    FlashRank (local cross-         |
        |    encoder): 10 -> 3               |
        +-----------------+------------------+
                          |
                          v
        SSE tokens + sources + room cards to the browser
```

Indexing happens once at startup: `hotel_data.json` becomes 29 chunks stored in Qdrant, then reused for every request.

## Decisions worth knowing

- AI writes the prose, code does the math. Room filtering, night counts, and totals live in `backend/app/core/tools.py` so prices cannot be hallucinated.
- Grounding is enforced twice: the system prompt only allows facts from the retrieved chunks, and every reply carries its sources. No relevant chunk means a concierge fallback, not a guess.
- Local models were chosen for cost, not just principle. Embeddings and reranking run on CPU with no quotas, which keeps the hosted demo free.
- The frontend never calls the model or stores keys. All AI traffic goes through the backend.

## Tests

111 automated backend tests (retrieval, reranking, tools, guard, API, streaming):

```bash
cd backend
uv run pytest tests/ -q
```

`backend/scripts/eval.py` adds 10 end-to-end scenarios: normal questions, missing dates, ambiguous queries, tool calls, unsupported assumptions, follow-ups, and failure fallbacks.

## Built with

Antigravity, Claude, Opencode.