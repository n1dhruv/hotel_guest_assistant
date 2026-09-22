# System Architecture: AI-Powered Hotel Guest Assistant

## 1. Overview & High-Level Architecture

The **Hotel Guest Assistant** is a full-stack, enterprise-grade conversational application designed for hotel guests. It resolves everyday guest inquiries (property info, amenities, policies, FAQs) and performs deterministic room availability checks with real-time pricing and room options.

```
                              ┌──────────────────────────────────────────────┐
                              │            Next.js 14 Web Frontend          │
                              │  (Chat Stream, Date Picker, Room Cards)      │
                              └──────────────────────┬───────────────────────┘
                                                     │ HTTP / REST & SSE Stream
                                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FastAPI Backend Service                                     │
│                                                                                             │
│  1. Ingress & Guard                                                                         │
│     POST /api/chat ──► [Prompt Injection Guard] ──► Passes or triggers security fallback    │
│                                                                                             │
│  2. RAG Retrieval Layer                                                                     │
│     Query ──► [In-Memory RAG Retriever] ──► Top-K Cosine Similar Knowledge Chunks           │
│                                                                                             │
│  3. LangChain Orchestration & Decisioning                                                   │
│     Retrieved Chunks + System Prompt + Date Anchor ──► LLM (OpenAI / Claude / Mock)          │
│                                                     │                                       │
│                                       Decision: Need Tool Call?                             │
│                                                     ├──► NO: Natural language response      │
│                                                     └──► YES: tool_call: checkAvailability  │
│                                                                                             │
│  4. Deterministic Execution Layer                                                           │
│     Tool Arguments ──► [Validation & checkAvailability()]                                   │
│                        (Date logic, capacity filter, inventory check, pricing calc)         │
│                        │                                                                    │
│                        └─► Returns JSON result back to LLM / Frontend UI                    │
│                                                                                             │
│  5. Observability & Telemetry                                                               │
│     [StatsTracker] ──► Records latency, tool call rate, fallback frequency                  │
│     GET /api/stats ──► Exposes real-time system performance & usefulness metrics            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Deterministic vs. AI Separation of Concerns

A critical requirement is that deterministic business logic must never be left to the probabilistic whims of an LLM.

| Responsibility | Handled By | Rationale |
| :--- | :--- | :--- |
| **User Intent Understanding** | **AI (LLM)** | Natural language variations ("Any room for me & my wife next weekend?") require semantic comprehension. |
| **Tool Argument Extraction** | **AI (LLM)** | Extracts check-in date, check-out date, and guest count from unstructured text. |
| **Availability Lookup** | **Deterministic (Python)** | Room inventory, date arithmetic, and availability status must be 100% accurate and auditable. |
| **Pricing & Night Calculations** | **Deterministic (Python)** | Mathematical calculations (price × nights) must never be hallucinated. |
| **Knowledge Base Grounding** | **Deterministic (RAG)** | Chunks are retrieved via cosine similarity from verified hotel data, preventing ungrounded claims. |
| **Prompt Injection Protection**| **Deterministic (Regex Guard)** | Fast, deterministic regex rules catch adversarial jailbreak patterns before hitting the model. |
| **Final Answer Synthesis** | **AI (LLM)** | Converts raw structured results and hotel facts into a polite, warm hospitality response. |

---

## 3. Data Flow Step-by-Step

1. **Guest Input**: Guest enters a question or clicks an availability prompt in the Next.js UI.
2. **Security Gate**: `guard.py` scans for jailbreak attempts (`"ignore instructions"`, `"system prompt"`). If detected, an immediate polite refusal is returned.
3. **Semantic Retrieval (RAG-Lite)**: The question is embedded and compared via cosine similarity against pre-chunked hotel data (`hotel_data.json`). The top 3–4 chunks are retrieved.
4. **Prompt Assembly**: The system prompt is constructed dynamically with:
   - Current date anchor (`YYYY-MM-DD (Day)`).
   - Retrieved hotel facts.
   - Strict operating instructions: answer only from context, trigger fallback if unknown, do not guess missing booking dates.
5. **Tool Decision & Invocation**:
   - If the guest asks about amenities or policies, the LLM answers directly from retrieved context.
   - If the guest asks for availability with dates & guest count, the LLM emits a tool call: `check_room_availability(checkIn, checkOut, adults)`.
   - If details are missing, the LLM politely prompts for the missing check-in/out dates or guests.
6. **Tool Execution**: Python executes `check_availability()`: validates dates (check-out > check-in), filters rooms by `maxGuests >= adults`, calculates total price, and returns structured data.
7. **Response & UI Render**: The frontend receives both the conversational answer and structured room cards to display visually.
8. **Observability Log**: Request latency, tool usage, and fallback status are recorded in `StatsTracker`.

---

## 4. Anti-Hallucination & Fallback Strategy

To ensure zero hallucinations:
1. **Strict Negative Constraint**: The model prompt explicitly orders: *"If information is not provided in context, do NOT speculate. State that you do not have that information and direct the guest to the front desk."*
2. **Date Anchoring**: LLMs often hallucinate when given relative dates ("tomorrow", "next Friday"). Passing the server's current date anchors all date resolution accurately.
3. **Structured Fallback Triggers**:
   - Unanswerable hotel question ➔ Polite fallback with concierge phone and email.
   - Invalid dates (e.g. check-out before check-in) ➔ Clear error message explaining the issue.
   - Capacity exceeded (>5 guests) ➔ Suggests contacting group reservations for connecting suites.
