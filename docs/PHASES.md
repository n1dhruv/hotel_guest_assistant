# Hotel Guest Assistant — Phased Implementation Plan

This document outlines the detailed step-by-step roadmap for building the **AI-Powered Hotel Guest Assistant**. Each phase is self-contained with clear deliverables, verification criteria, and zero ambiguity.

---

## Summary of Phases

| Phase | Title | Core Focus | Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Scaffolding & Environment Setup** | Project setup & dependency installation | `backend/requirements.txt`, `backend/.venv`, `frontend/` Next.js app, `.env.example` |
| **Phase 2** | **Knowledge Base & Deterministic Tools** | Ground truth data & availability logic | `hotel_data.json`, `tools.py` (`check_availability`), input validation |
| **Phase 3** | **Security Guard & RAG-Lite Retrieval** | Adversarial safety & semantic search | `guard.py` (injection detector), `rag.py` (in-memory cosine retrieval) |
| **Phase 4** | **LangChain Agent & Core Chat API** | AI orchestration & tool binding | `chain.py` (LangChain + prompt + tools), `api/chat.py`, `main.py` |
| **Phase 5** | **Observability & Telemetry API** | Metrics & usefulness tracking | `stats.py` (`StatsTracker`), `api/stats.py` (`GET /api/stats`) |
| **Phase 6** | **Frontend Experience (Next.js)** | Responsive chat UI & interactive cards | `ChatInterface.tsx`, `AvailabilityCard.tsx`, `DateGuestPicker.tsx` |
| **Phase 7** | **Automated Backend Test Suite** | Reliability & unit verification | `tests/test_availability.py`, `tests/test_guard.py`, `tests/test_api.py` |
| **Phase 8** | **Golden Evaluation Harness (8–10 Scenarios)** | Empirical proof & usefulness grading | `scripts/eval.py`, `docs/EVALUATION.md` (pass/fail table) |
| **Phase 9** | **Documentation & Submission Package** | Production write-up & README | `README.md`, `docs/AI_DECISIONS.md`, clean Git commit |

---

## Detailed Phase Breakdown

### Phase 1: Scaffolding & Environment Setup
- **Objective**: Establish clean, reproducible development environments for backend (and frontend).
- **Key Tasks**:
  1. Initialize Python 3.12 virtual environment in `backend/.venv` using `uv`.
  2. Define and install backend dependencies (`fastapi`, `uvicorn`, `langchain`, `langchain-core`, `langchain-openai`, `pydantic-settings`, `pytest`, `httpx`, `numpy`).
  3. Create backend configuration schema (`backend/app/config.py`).
  4. Create `.env.example` documenting required configuration (`OPENAI_API_KEY`, `MOCK_LLM`, etc.).
- **Verification**:
  - `backend/.venv/bin/python -c "import fastapi, langchain, pydantic; print('BACKEND_SETUP_OK')"` exits with code 0.

---

### Phase 2: Hotel Knowledge Base & Deterministic Availability Tool
- **Objective**: Establish the hotel ground truth data and build the core deterministic availability tool.
- **Key Tasks**:
  1. Create `backend/app/data/hotel_data.json` containing:
     - Property details (name, check-in 3:00 PM, check-out 11:00 AM, address, phone).
     - 6 amenities (pool, gym, complimentary breakfast, Wi-Fi, parking, restaurant).
     - 4 room tiers (Standard Queen, Deluxe King, Family Suite, Presidential Penthouse).
     - 6 hotel policies (cancellation, check-in/out, pet policy, children, smoking, payment).
     - 8 curated FAQs.
  2. Implement `backend/app/core/tools.py`:
     - Function: `check_availability(checkIn: str, checkOut: str, adults: int = 2) -> dict`
     - Robust date parsing (`YYYY-MM-DD`, etc.) and validation (`checkOut > checkIn`).
     - Capacity checking: filters rooms where `maxGuests >= adults`.
     - Pricing calculation: `nights * basePricePerNight`.
     - Deterministic availability mock rule (clearly documented).
- **Verification**:
  - Run test verification on valid date ranges, invalid dates, capacity overflows, and pricing calculations.

---

### Phase 3: Security Guard & RAG-Lite Retrieval Layer
- **Objective**: Protect the system against prompt injection and build an in-memory RAG retriever.
- **Key Tasks**:
  1. Implement `backend/app/core/guard.py`:
     - Detects adversarial patterns (`"ignore previous instructions"`, `"reveal system prompt"`, `"you are now"`, `"DAN mode"`).
     - Returns `(is_suspicious, reason)`.
  2. Implement `backend/app/core/rag.py`:
     - Parses `hotel_data.json` into structured, semantic chunks.
     - Dual-mode embedding & retrieval: dense vector search if OpenAI credentials are set, with a zero-dependency TF-IDF cosine-similarity fallback for 100% offline reliability.
     - Method `retrieve(query: str, k: int = 4)` returns top-k matching hotel facts.
- **Verification**:
  - Run queries for "pool" and "cancellation" to verify top-ranked chunks match the exact policy.
  - Verify injection strings are correctly flagged.

---

### Phase 4: LangChain Agent & Core Chat API
- **Objective**: Wire AI orchestration, tool binding, and the FastAPI chat endpoint.
- **Key Tasks**:
  1. Implement `backend/app/core/chain.py`:
     - Bind `check_room_availability` tool to the model.
     - Dynamic system prompt with real-time Date Anchor (`Today's Date: YYYY-MM-DD`) and retrieved RAG context.
     - Dual execution engine: Live LLM (`gpt-4o-mini`) + high-fidelity Mock Engine fallback so the app works even with no API keys.
  2. Implement `backend/app/api/chat.py`:
     - `POST /api/chat`: Accepts conversation history `{ messages: [...] }`.
     - Runs security check ➔ RAG retrieval ➔ LangChain tool execution ➔ response formatting.
  3. Implement `backend/app/main.py`:
     - FastAPI entrypoint with CORS middleware for frontend connection.
- **Verification**:
  - Send `curl` requests for a normal FAQ and an availability request; receive valid structured responses.

---

### Phase 5: Observability & Telemetry API
- **Objective**: Track and expose operational metrics answering *"How would you measure usefulness?"*.
- **Key Tasks**:
  1. Implement `backend/app/core/stats.py`:
     - `StatsTracker` recording request volume, tool call count, fallback triggers, injection blocks, and response latencies.
  2. Implement `backend/app/api/stats.py`:
     - `GET /api/stats`: Returns live JSON metrics:
       ```json
       {
         "total_requests": 24,
         "tool_call_rate": 0.25,
         "fallback_rate": 0.08,
         "avg_latency_ms": 320.5,
         "status": "healthy"
       }
       ```
- **Verification**:
  - `curl http://localhost:8000/api/stats` returns up-to-date metrics after chat calls.

---

### Phase 6: Frontend Experience (Next.js & Tailwind)
- **Objective**: Create a polished, responsive guest-facing web interface.
- **Key Tasks**:
  1. `frontend/src/components/ChatInterface.tsx`:
     - Message thread with user and assistant bubbles.
     - Loading / typing animation and quick-start prompt chips.
     - Graceful error banners with retry buttons.
  2. `frontend/src/components/AvailabilityCard.tsx`:
     - Visual presentation of returned rooms: image/icon, badge, price per night, total stay price, max guests.
  3. `frontend/src/components/DateGuestPicker.tsx`:
     - Inline widget to select check-in date, check-out date, and guest count when availability is needed.
  4. Fully responsive layout on mobile, tablet, and desktop.
- **Verification**:
  - Open `http://localhost:3000`, test asking questions, asking follow-ups, and triggering availability cards.

---

### Phase 7: Automated Backend Test Suite (pytest)
- **Objective**: Meet the brief's requirement for *"meaningful automated tests for important backend flows"*.
- **Key Tasks**:
  1. `backend/tests/test_availability.py`:
     - Tests normal booking, checkout-before-checkin, negative guest count, large group limits.
  2. `backend/tests/test_guard.py`:
     - Tests various prompt injection strings and validates clean inputs pass.
  3. `backend/tests/test_api.py`:
     - Tests `POST /api/chat`, `GET /api/stats`, and `POST /api/availability` using `TestClient`.
- **Verification**:
  - Run `pytest backend/tests` and achieve 100% pass rate.

---

### Phase 8: Golden Evaluation Harness (8–10 Scenarios)
- **Objective**: Execute and document the 8–10 required evaluation scenarios.
- **Key Tasks**:
  1. Implement `backend/scripts/eval.py`:
     - Executes 10 predefined real-world scenarios covering normal FAQ, policy questions, room fit, availability with complete info, missing dates, ambiguous queries, unsupported questions, follow-ups, injection attempts, and e2e flow.
     - Evaluates response against expected assertions (e.g. tool called, fallback triggered, correct details present).
     - Prints a formatted pass/fail summary table.
  2. Save output to `docs/EVALUATION.md`.
- **Verification**:
  - Run `python backend/scripts/eval.py` and verify all 10 scenarios pass with documented evidence.

---

### Phase 9: Documentation & Submission Package
- **Objective**: Complete all deliverables requested by the assignment.
- **Key Tasks**:
  1. Write root `README.md` with setup/run instructions, API curl examples, and architecture summary.
  2. Write `docs/AI_DECISIONS.md`:
     - Customer problem & guest journey.
     - UX design rationale (chat vs inline forms).
     - Deterministic vs AI boundary defense.
     - Hallucination prevention and failure modes.
     - Usefulness measurement and production roadmap.
- **Verification**:
  - Review all documentation against the assignment PDF checklist.
