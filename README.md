# Hotel Guest Assistant

A full-stack guest assistant web application built for The Grand Azure Heritage Resort & Spa in Candolim, Goa. The system handles guest inquiries about amenities, policies, dining, and room bookings using Retrieval-Augmented Generation (RAG) alongside a deterministic room availability engine.

---

## What the Project Does

When guests interact with the hotel assistant, the system performs several tasks:

1. **Answers property questions**: Explains amenities, check-in and check-out timings, dining options, ID verification requirements, cancellation terms, and general house rules using the verified resort knowledge base.
2. **Checks live room availability**: When given check-in dates, check-out dates, and guest counts, it calculates stay length, verifies guest capacity per room type, calculates tariffs, and returns structured room cards.
3. **Clarifies missing booking details**: If a guest asks about room availability without providing dates, the assistant prompts them for the missing information and presents an interactive date picker.
4. **Handles out-of-scope inquiries**: If a guest asks about external sightseeing, water sports not offered on site, or unrelated topics, the system avoids inventing details and directs them to the concierge desk with verified contact details.
5. **Defends against prompt injections**: Scans incoming messages for jailbreak attempts, developer prompt extraction, and instructions to override safety guidelines.

---

## How the System Works

```
Guest Input (Browser)
       |
       v
[FastAPI Backend: /api/chat]
       |
       +---> [1. Prompt Injection Gate] ---> Block if malicious
       |
       +---> [2. RAG Knowledge Retrieval] ---> Top matching chunks from hotel_data.json
       |
       +---> [3. Orchestrator]
                 |
                 |--> If live LLM enabled: Calls model via LiteLLM with tool-calling
                 |
                 |--> If mock/offline mode: Deterministic RAG engine
                          |
                          |--> Availability check (if dates provided)
                          |--> Topic-deduplicated answer synthesis
       |
       +---> [4. Response Delivery]
                 |--> Structured JSON with text reply, sources, and room cards
                 |--> Real-time typewriter streaming on the frontend
```

### 1. Security Guard
Every message passes through `app/core/guard.py` before hitting any language model or retrieval index. It uses regex patterns covering common prompt injections, system prompt exfiltration attempts, developer mode triggers, and chat template token manipulations. If flagged, the request is immediately neutralized with a polite boundary message.

### 2. Knowledge Retrieval (RAG)
Hotel ground truth is stored in `backend/app/data/hotel_data.json` covering:
- Property overview and contact details
- Resort amenities (pool, spa, gym, breakfast, parking, dining)
- Room tiers, capacities, and base rates
- Policies (cancellation, check-in/out, ID verification, pets, children, smoking, payments)
- Curated guest FAQs

At startup, `app/core/rag.py` splits this data into semantic chunks and indexes them with an in-memory BM25 lexical retriever. The retriever applies custom text normalization (such as unifying "check-in" to "checkin"), stopword filtering, morphological suffix stemming, title weighting, and synonym expansion. If an OpenAI API key is supplied, dense vector embeddings (`text-embedding-3-small`) can optionally be used instead.

### 3. Answer Synthesis and Deduplication
In the hotel dataset, information often appears in multiple formats. For example, the swimming pool is documented both as an amenity catalog item and as a conversational FAQ.

To prevent redundant duplicate answers:
- Each chunk has an assigned canonical topic (`pool`, `breakfast`, `checkin`, `checkout`, `cancellation`, etc.).
- When resolving specific questions, candidate chunks are grouped by topic, and the system selects the single most conversational chunk (preferring FAQ answers over raw catalog entries).
- For compound questions (such as asking for both check-in and check-out times in a single sentence), both topics are recognized and answered together.
- For broad questions (such as asking for all amenities or an overview of the resort), dedicated synthesis functions format comprehensive, categorized listings from the entire knowledge base.

### 4. Room Availability Engine
Availability checks run through `app/core/tools.py`. The tool:
- Validates that check-in occurs today or in the future, and check-out is after check-in.
- Calculates the total nights of stay.
- Filters room tiers by maximum guest capacity.
- Applies seasonal pricing multipliers (peak holiday seasons vs standard dates).
- Calculates the total stay cost for each eligible room type.
- Returns structured JSON used by the frontend to render interactive room cards.

### 5. Frontend Experience
The interface is built with Next.js and Tailwind CSS:
- **Typewriter streaming**: Rather than waiting for a full payload, the response reveals progressively with a blinking cursor at human reading speed.
- **Structured cards**: When room availability is checked, interactive room tier cards display prices, bedding details, and capacity badges.
- **Date picker**: Appears automatically whenever the assistant detects that booking dates were omitted.
- **Quick prompts**: Pre-set buttons for common queries like check-in times, pool hours, cancellation policies, and breakfast.

---

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, Uvicorn, LiteLLM, NumPy
- **Frontend**: Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS, Lucide Icons
- **Package Managers**: `uv` for Python virtual environment and dependencies, `npm` for Node.js
- **Testing**: Pytest with pytest-asyncio

---

## Project Structure

```
.
|-- backend/
|   |-- app/
|   |   |-- api/
|   |   |   |-- chat.py            # Chat and streaming endpoints
|   |   |   |-- availability.py    # Direct room availability endpoint
|   |   |   |-- stats.py           # Operational telemetry endpoint
|   |   |-- core/
|   |   |   |-- chain.py           # Universal orchestrator & synthesis logic
|   |   |   |-- guard.py           # Prompt injection security firewall
|   |   |   |-- rag.py             # In-memory BM25 & dense vector store
|   |   |   |-- tools.py           # Room availability & pricing calculator
|   |   |   |-- stats.py           # Telemetry metrics tracker
|   |   |-- data/
|   |   |   |-- hotel_data.json    # Verified hotel knowledge base
|   |   |-- config.py              # Environment configuration via Pydantic
|   |   |-- main.py                # FastAPI application entrypoint
|   |-- tests/                     # 27 automated unit and integration tests
|   |-- pyproject.toml             # Python package configuration
|   |-- .env                       # Backend environment settings
|-- frontend/
|   |-- src/
|   |   |-- app/                   # Next.js App Router pages and layout
|   |   |-- components/
|   |   |   |-- ChatInterface.tsx  # Main interactive chat console
|   |   |   |-- AvailabilityCard.tsx # Room inventory & pricing display
|   |   |   |-- DateGuestPicker.tsx  # Inline booking dates selector
|   |   |   |-- FormattedMessage.tsx # Markdown renderer for assistant messages
|   |-- package.json
|-- docs/                          # Architecture decisions and evaluation reports
|-- README.md
```

---

## Getting Started

### Prerequisites
- Python 3.12 or newer with `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Node.js 18 or newer with `npm`

### 1. Run the Backend

From the repository root:

```bash
cd backend
uv run backend
```

The API server starts at `http://localhost:8000`. You can test health by visiting `http://localhost:8000/health` or view the interactive OpenAPI documentation at `http://localhost:8000/docs`.

### 2. Run the Frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your web browser.

---

## Configuration

Settings are controlled via `backend/.env`. A template is provided in `backend/.env.example`.

```env
# LiteLLM Model Selection (e.g. gpt-4o-mini, gemini/gemini-2.5-flash, groq/llama-3.3-70b-versatile, ollama/llama3)
LLM_MODEL=gpt-4o-mini

# Provider API Keys (leave blank to run in offline dynamic RAG mode)
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
GROQ_API_KEY=

# Force deterministic RAG mode even if keys are present
MOCK_LLM=true

# Server configuration
DEBUG=true
PORT=8000
HOST=0.0.0.0
```

### Running with a Live LLM vs Mock Mode
- **Offline / Mock Mode (`MOCK_LLM=true`)**: Runs entirely on the local machine with no external network calls. Retrieval, answer synthesis, topic deduplication, and tool calls are handled deterministically in sub-millisecond response times.
- **Live LLM Mode (`MOCK_LLM=false`)**: Passes retrieved context and tool definitions to your chosen model via LiteLLM. If the external provider experiences network latency or fails, the orchestrator automatically falls back to the local RAG engine.

---

## API Endpoints

### `POST /api/chat`
Main conversational endpoint. Accepts conversation history and returns the assistant's reply along with verified source references and availability data if tool calling was triggered.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What time is check-in and check-out?"}
  ]
}
```

**Response:**
```json
{
  "reply": "Standard check-in begins at 2:00 PM IST. Early check-in starting from 10:00 AM can be arranged subject to room availability.\n\nCheck-out time is by 11:00 AM IST. Late check-out until 2:00 PM may be requested at the front desk, subject to availability.",
  "tool_called": false,
  "availability": null,
  "used_fallback": false,
  "injection_blocked": false,
  "needs_dates": false,
  "retrieved_sources": ["FAQ: What time is check-in?", "FAQ: What time is check-out?"],
  "latency_ms": 0.8
}
```

### `POST /api/chat/stream`
Server-Sent Events (SSE) streaming endpoint that emits tokens in real-time as they are produced, followed by a metadata event containing retrieved sources and availability results.

### `POST /api/availability`
Direct endpoint for querying room rates and capacity without going through the chat assistant.

**Request:**
```json
{
  "checkIn": "2026-10-15",
  "checkOut": "2026-10-18",
  "adults": 2
}
```

### `GET /api/stats`
Telemetry endpoint that reports real-time metrics including total requests, tool call rates, injection blocks, and average latency.

---

## Testing

The project includes an automated test suite covering security checks, RAG retrieval accuracy, availability calculations, and API routes.

Run the tests from the `backend` directory:

```bash
cd backend
uv run pytest
```

All 27 test cases run against the local test suite without requiring third-party API credentials.
