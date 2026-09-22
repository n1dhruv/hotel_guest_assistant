# Hotel Guest Assistant — Master Project Plan

## Executive Summary
This project builds a full-stack, AI-powered **Hotel Guest Assistant** web application for **The Grand Azure Hotel & Suites**. It allows website visitors to ask hotel-related questions and deterministically check room availability with transparent room options and pricing.

---

## Technical Stack Selection

| Component | Selected Technology | Rationale |
| :--- | :--- | :--- |
| **Frontend** | **Next.js 14/15 (App Router, TypeScript, Tailwind CSS)** | Clean, responsive UI with fast SSR, type safety, and seamless mobile responsiveness. |
| **Backend** | **FastAPI (Python 3.12)** | Asynchronous, high-performance REST API with native Pydantic schema validation. |
| **AI Orchestration** | **LangChain (`langchain-core`, `langchain-openai`)** | Industry-standard tool binding (`bind_tools`), prompt templating, and provider agnosticism. |
| **Retrieval Layer** | **In-Memory RAG-Lite (Cosine Similarity)** | Embeds hotel data chunks into vectors; performs top-k cosine similarity retrieval. Dual-mode support (OpenAI dense vectors + offline TF-IDF fallback). |
| **Deterministic Tool** | **Python `check_availability()`** | Strict mathematical calculations (date validation, room capacity, nights, total price) kept 100% outside the LLM. |
| **Security Layer** | **Regex Guard (`guard.py`)** | Detects and blocks prompt injection and jailbreak attempts before they reach the model. |
| **Testing & Evaluation** | **pytest + automated `eval.py` harness** | Backend unit tests + automated runner testing 10 golden evaluation scenarios with pass/fail grading. |

---

## Documentation Index

The following detailed documentation is available in the [`docs/`](./) folder:

1. **[PHASES.md](./PHASES.md)**:
   - Full 9-phase implementation roadmap.
   - For every phase: exact inputs, files to create/edit, verification commands, and acceptance criteria.
2. **[ARCHITECTURE.md](./ARCHITECTURE.md)**:
   - End-to-end component diagram and data flow.
   - Deterministic vs AI split explanation.
   - Anti-hallucination, grounding, and security guardrail strategies.
3. **[EVALUATION_PLAN.md](./EVALUATION_PLAN.md)**:
   - Detailed specification of the 10 golden evaluation scenarios covering normal questions, edge cases, missing dates, ambiguities, prompt injection, and end-to-end booking.

---

## Phase Roadmap at a Glance

- **Phase 1: Scaffolding & Environment Setup (Backend & Frontend)**
- **Phase 2: Hotel Knowledge Base & Deterministic Tools** (`hotel_data.json`, `check_availability`)
- **Phase 3: Security Guard & RAG-Lite Retrieval** (`guard.py`, `rag.py`)
- **Phase 4: LangChain Agent & Core Chat API** (`chain.py`, `api/chat.py`, `main.py`)
- **Phase 5: Observability & Telemetry API** (`stats.py`, `GET /api/stats`)
- **Phase 6: Frontend Experience (Next.js)** (Chat UI, room cards, date-guest selector)
- **Phase 7: Automated Backend Test Suite** (pytest unit & integration tests)
- **Phase 8: Golden Evaluation Harness** (`eval.py`, real observed results table)
- **Phase 9: Documentation & Submission Package** (README, AI decisions write-up, git clean)
