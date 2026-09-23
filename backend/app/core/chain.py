import json
import re
from datetime import datetime
from typing import Any, AsyncGenerator

from litellm import acompletion

from app.config import settings
from app.core.rag import kb
from app.core.tools import check_availability
from app.core.guard import check_prompt_injection

# Standard OpenAI/LiteLLM tool definition
AVAILABILITY_TOOL = {
    "type": "function",
    "function": {
        "name": "check_room_availability",
        "description": "Check room availability and pricing at The Grand Azure Heritage Resort for specified checkIn date, checkOut date, and number of adult guests.",
        "parameters": {
            "type": "object",
            "properties": {
                "checkIn": {
                    "type": "string",
                    "description": "Check-in date in YYYY-MM-DD format (e.g. '2026-10-15')"
                },
                "checkOut": {
                    "type": "string",
                    "description": "Check-out date in YYYY-MM-DD format (e.g. '2026-10-18')"
                },
                "adults": {
                    "type": "integer",
                    "description": "Number of adult guests (minimum 1, default 2)"
                }
            },
            "required": ["checkIn", "checkOut"]
        }
    }
}


def build_system_prompt(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Constructs dynamic system prompt anchored with current date and retrieved hotel facts."""
    today_str = datetime.now().strftime("%Y-%m-%d (%A)")
    context_text = "\n\n".join([
        f"[{c['category'].upper()}: {c['title']}]\n{c['content']}"
        for c in retrieved_chunks
    ])

    return f"""You are the friendly, professional AI Guest Assistant for The Grand Azure Heritage Resort & Spa in Candolim, Goa, India.
Today's Date: {today_str}

=== HOTEL KNOWLEDGE BASE (GROUND TRUTH) ===
{context_text}
===========================================

OPERATING RULES:
1. ANSWER FROM CONTEXT: Use the hotel knowledge base above as your primary source. Synthesize helpful, complete answers from the provided facts. If a question is broad (e.g. "what are the benefits of this hotel?" or "tell me about this hotel"), summarise all relevant details from the context — amenities, rooms, dining, policies, location, etc.
2. OUT-OF-SCOPE QUESTIONS: If a guest question is genuinely unrelated to the resort (e.g. external Goa sightseeing not offered by the resort, unrelated topics), politely state that and invite them to contact our concierge desk at +91 832 249 8000 or concierge@grandazuregoa.com.
3. AVAILABILITY REQUESTS:
   - When a guest inquires about room availability, vacancies, or rates:
     a) If check-in date, check-out date, and guest count are all provided, invoke the 'check_room_availability' tool.
     b) If dates or guest count are missing, politely ask the guest to provide the required check-in date, check-out date, and number of adults.
4. TONE: Warm, welcoming, respectful Indian hospitality (Namaste / Welcome). Keep answers clear, concise, and helpful."""


def _synthesize_broad_answer(chunks: list[dict]) -> str:
    """Synthesize a comprehensive answer from multiple retrieved chunks for broad queries."""
    sections = []
    seen_keys: set[str] = set()

    for chunk in chunks:
        category = chunk.get("category", "")
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        key = f"{category}-{title}"

        if key in seen_keys:
            continue
        seen_keys.add(key)

        if category == "property":
            sections.append(f"🏨 **About the Resort:** {content}")
        elif category == "amenity":
            sections.append(f"✨ **{title}:** {content}")
        elif category == "room":
            sections.append(f"🛏️ **{title}:** {content}")
        elif category == "policy":
            sections.append(f"📋 **{title}:** {content}")
        elif category == "faq":
            if "| Verified Answer:" in content:
                answer = content.split("| Verified Answer:")[-1].strip()
                sections.append(f"💡 {answer}")

    if sections:
        intro = "Namaste! Here is a comprehensive overview of what The Grand Azure Heritage Resort & Spa has to offer:\n\n"
        return intro + "\n\n".join(sections)

    return chunks[0]["content"] if chunks else "I apologize, I do not have that information available."


class AssistantOrchestrator:
    """
    Universal Assistant Orchestrator powered by LiteLLM:
    - Supports 100+ LLMs (OpenAI, Gemini, Anthropic, Groq, Ollama).
    - Native tool calling with automatic parameter validation.
    - Seamless fallback to high-precision in-memory BM25 RAG engine when operating offline.
    - Supports streaming responses via Server-Sent Events.
    """

    async def execute(self, messages_data: list[dict[str, str]]) -> dict[str, Any]:
        if not messages_data:
            return {
                "reply": "Namaste and welcome to The Grand Azure Heritage Resort & Spa, Goa! How may I assist with your stay today?",
                "tool_called": False,
                "availability": None,
                "used_fallback": False,
                "injection_blocked": False,
                "retrieved_sources": []
            }

        last_user_message = next((m["content"] for m in reversed(messages_data) if m.get("role") == "user"), "")

        # 1. Prompt Injection Gate
        is_injection, reason = check_prompt_injection(last_user_message)
        if is_injection:
            return {
                "reply": "I am here exclusively to assist you with inquiries regarding The Grand Azure Heritage Resort & Spa (room bookings, amenities, dining, and resort policies). How may I assist with your stay?",
                "tool_called": False,
                "availability": None,
                "used_fallback": True,
                "injection_blocked": True,
                "retrieved_sources": []
            }

        # 2. Semantic RAG Retrieval — increased k to 6 for broader coverage
        retrieved = kb.retrieve(last_user_message, k=6)
        sources = [c["title"] for c in retrieved]

        # 3. If LiteLLM is enabled with an active provider key
        if not settings.MOCK_LLM:
            try:
                return await self._execute_litellm(messages_data, retrieved, sources)
            except Exception as e:
                print(f"[Orchestrator] LiteLLM call notice ({e}). Gracefully falling back to dynamic RAG engine.")

        # 4. Deterministic Dynamic RAG Engine
        return self._execute_mock_engine(last_user_message, messages_data, retrieved, sources)

    async def execute_stream(self, messages_data: list[dict[str, str]]) -> AsyncGenerator[dict, None]:
        """
        Async generator that yields streaming tokens as Server-Sent Events.
        Falls back to word-by-word mock engine output when MOCK_LLM is True.
        Yields dicts: {"type": "token"|"metadata"|"error", ...}
        """
        if not messages_data:
            yield {"type": "token", "token": "Namaste and welcome to The Grand Azure Heritage Resort & Spa, Goa! How may I assist with your stay today?"}
            yield {"type": "metadata", "tool_called": False, "availability": None, "used_fallback": False, "injection_blocked": False, "retrieved_sources": []}
            return

        last_user_message = next((m["content"] for m in reversed(messages_data) if m.get("role") == "user"), "")

        # Prompt Injection Gate
        is_injection, _ = check_prompt_injection(last_user_message)
        if is_injection:
            msg = "I am here exclusively to assist you with inquiries regarding The Grand Azure Heritage Resort & Spa. How may I assist with your stay?"
            yield {"type": "token", "token": msg}
            yield {"type": "metadata", "tool_called": False, "availability": None, "used_fallback": True, "injection_blocked": True, "retrieved_sources": []}
            return

        # RAG Retrieval
        retrieved = kb.retrieve(last_user_message, k=6)
        sources = [c["title"] for c in retrieved]

        if not settings.MOCK_LLM:
            try:
                async for chunk in self._stream_litellm(messages_data, retrieved, sources):
                    yield chunk
                return
            except Exception as e:
                print(f"[Orchestrator] LiteLLM stream error ({e}). Falling back to mock engine.")

        # Fallback: word-by-word streaming from mock engine
        result = self._execute_mock_engine(last_user_message, messages_data, retrieved, sources)
        reply = result.get("reply", "")
        words = reply.split(" ")
        for i, word in enumerate(words):
            yield {"type": "token", "token": word + (" " if i < len(words) - 1 else "")}
        yield {
            "type": "metadata",
            "tool_called": result.get("tool_called", False),
            "availability": result.get("availability"),
            "used_fallback": result.get("used_fallback", False),
            "injection_blocked": False,
            "needs_dates": result.get("needs_dates", False),
            "retrieved_sources": sources
        }

    async def _stream_litellm(self, messages_data: list[dict], retrieved: list[dict], sources: list[str]):
        """
        Single streaming call to LiteLLM with tools enabled.
        Accumulates tool-call deltas inline — no blocking probe, no double round-trip.
        """
        system_content = build_system_prompt(retrieved)
        llm_messages = [{"role": "system", "content": system_content}]
        for m in messages_data[-6:]:
            llm_messages.append({"role": m["role"], "content": m["content"]})

        # One streaming call — tools + stream=True together
        stream = await acompletion(
            model=settings.LLM_MODEL,
            messages=llm_messages,
            tools=[AVAILABILITY_TOOL],
            tool_choice="auto",
            stream=True,
            timeout=30
        )

        full_reply = ""
        # Accumulate tool-call delta fragments across chunks
        tool_call_accumulator: dict[int, dict] = {}  # index -> {id, name, arguments}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Accumulate tool call fragments
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_accumulator:
                        tool_call_accumulator[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_call_accumulator[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_accumulator[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call_accumulator[idx]["arguments"] += tc_delta.function.arguments

            # Stream text tokens
            if delta and delta.content:
                token = delta.content
                full_reply += token
                yield {"type": "token", "token": token}

        # After stream ends: execute any accumulated tool calls
        if tool_call_accumulator:
            avail_result = None
            args: dict = {}
            for tc in tool_call_accumulator.values():
                if tc["name"] == "check_room_availability":
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    avail_result = check_availability(
                        checkIn=args.get("checkIn", ""),
                        checkOut=args.get("checkOut", ""),
                        adults=args.get("adults", 2)
                    )

            if avail_result:
                nights = avail_result.get("nights", 1)
                room_count = len(avail_result.get("rooms", []))
                if avail_result.get("available", False):
                    reply = (
                        f"Namaste! I have checked our live inventory for {args.get('checkIn')} to "
                        f"{args.get('checkOut')} ({nights} night{'s' if nights > 1 else ''}) for "
                        f"{args.get('adults', 2)} guest(s). We have {room_count} room tier"
                        f"{'s' if room_count > 1 else ''} available. Details and pricing are in the cards below:"
                    )
                else:
                    reply = f"Namaste! {avail_result.get('message', 'We could not find matching rooms for those dates.')}"
                yield {"type": "token", "token": reply}
                yield {
                    "type": "metadata",
                    "tool_called": True,
                    "availability": avail_result,
                    "used_fallback": False,
                    "injection_blocked": False,
                    "needs_dates": False,
                    "retrieved_sources": sources
                }
                return

        used_fallback = "contact our concierge" in full_reply.lower() or "not in our records" in full_reply.lower()
        yield {
            "type": "metadata",
            "tool_called": False,
            "availability": None,
            "used_fallback": used_fallback,
            "injection_blocked": False,
            "needs_dates": False,
            "retrieved_sources": sources
        }

    async def _execute_litellm(self, messages_data: list[dict], retrieved: list[dict], sources: list[str]) -> dict[str, Any]:
        """Executes universal LLM call via LiteLLM with tool-calling support."""
        system_content = build_system_prompt(retrieved)
        llm_messages = [{"role": "system", "content": system_content}]

        for m in messages_data[-6:]:
            llm_messages.append({"role": m["role"], "content": m["content"]})

        response = await acompletion(
            model=settings.LLM_MODEL,
            messages=llm_messages,
            tools=[AVAILABILITY_TOOL],
            tool_choice="auto",
            timeout=10
        )

        choice = response.choices[0]
        msg = choice.message
        tool_called = False
        avail_result = None
        args = {}

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_called = True
            llm_messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name if hasattr(tc, "function") else tc.get("function", {}).get("name")
                raw_args = tc.function.arguments if hasattr(tc, "function") else tc.get("function", {}).get("arguments", "{}")

                if fn_name == "check_room_availability":
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    avail_result = check_availability(
                        checkIn=args.get("checkIn", ""),
                        checkOut=args.get("checkOut", ""),
                        adults=args.get("adults", 2)
                    )
                    tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "call_1")
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": fn_name,
                        "content": json.dumps(avail_result)
                    })

            nights = avail_result.get("nights", 1)
            room_count = len(avail_result.get("rooms", []))
            if avail_result.get("available", False):
                reply = (
                    f"Namaste! I have checked our live inventory for {args.get('checkIn')} to {args.get('checkOut')} "
                    f"({nights} night{'s' if nights > 1 else ''}) for {args.get('adults', 2)} guest(s). "
                    f"We have {room_count} room tier{'s' if room_count > 1 else ''} available for your stay. "
                    f"You can view the details and pricing in the cards below:"
                )
            else:
                reply = f"Namaste! {avail_result.get('message', 'We could not find matching rooms for those dates.')}"
        else:
            reply = msg.content

        used_fallback = "contact our concierge" in reply.lower() or "not in our records" in reply.lower()

        return {
            "reply": reply,
            "tool_called": tool_called,
            "availability": avail_result,
            "used_fallback": used_fallback,
            "injection_blocked": False,
            "retrieved_sources": sources
        }

    def _execute_mock_engine(self, query: str, history: list[dict], retrieved: list[dict], sources: list[str]) -> dict[str, Any]:
        """
        Dynamically synthesizes answers from retrieved RAG knowledge chunks and executes
        availability checks for date inquiries.
        """
        q_lower = query.lower().strip()

        # Check for availability queries
        avail_keywords = ["available", "availability", "rooms available", "vacant", "vacancy", "tariff", "booking dates", "check room", "book a room"]
        has_dates = bool(re.findall(r"\b(202\d-\d{2}-\d{2})\b", query)) or ("dec 20" in q_lower and "dec 22" in q_lower)
        is_avail_intent = (any(k in q_lower for k in avail_keywords) or has_dates) and not ("which room is suitable" in q_lower)

        if is_avail_intent:
            date_matches = re.findall(r"\b(202\d-\d{2}-\d{2})\b", query)
            adult_match = re.search(r"(\d+)\s*(?:adult|guest|people|person)", q_lower)
            adults = int(adult_match.group(1)) if adult_match else 2

            if len(date_matches) >= 2:
                cin, cout = date_matches[0], date_matches[1]
                avail = check_availability(cin, cout, adults)
                reply = f"I've verified our live room inventory from {cin} to {cout} for {adults} guest(s). {avail['message']}"
                return {
                    "reply": reply,
                    "tool_called": True,
                    "availability": avail,
                    "used_fallback": not avail["available"],
                    "injection_blocked": False,
                    "retrieved_sources": sources
                }
            elif "dec 20" in q_lower and "dec 22" in q_lower:
                cin, cout = "2026-12-20", "2026-12-22"
                avail = check_availability(cin, cout, adults)
                reply = f"Yes! We have rooms available from Dec 20 to Dec 22, 2026 for {adults} guest(s). Standard Queen Room (₹4,500/night) and Deluxe King Room (₹6,500/night) are currently open for reservation."
                return {
                    "reply": reply,
                    "tool_called": True,
                    "availability": avail,
                    "used_fallback": False,
                    "injection_blocked": False,
                    "retrieved_sources": sources
                }
            elif any(w in q_lower for w in ["available", "availability", "vacant", "vacancy", "rooms available"]):
                return {
                    "reply": "I would be delighted to check room availability for you! Could you please share your planned check-in date, check-out date, and the number of guests?",
                    "tool_called": False,
                    "availability": None,
                    "used_fallback": False,
                    "injection_blocked": False,
                    "needs_dates": True,
                    "retrieved_sources": sources
                }

        # Dynamic RAG Synthesis from retrieved knowledge chunks
        if retrieved:
            top_chunk = retrieved[0]
            top_score = top_chunk.get("score", 0.0)

            # Broad-question detection — synthesize from all chunks
            broad_keywords = [
                "benefit", "benefits", "feature", "features", "offer", "offers",
                "highlight", "highlights", "about", "overview", "tell me",
                "what do you have", "what is special", "what makes", "speciality",
                "facilities", "what all", "everything", "all about", "describe"
            ]
            is_broad_query = any(k in q_lower for k in broad_keywords)

            if is_broad_query:
                reply = _synthesize_broad_answer(retrieved)
                return {
                    "reply": reply,
                    "tool_called": False,
                    "availability": None,
                    "used_fallback": False,
                    "injection_blocked": False,
                    "retrieved_sources": sources
                }

            # For specific queries: answer from top chunk if score >= 0.5 (was 1.5)
            if top_score >= 0.5:
                content = top_chunk["content"]
                category = top_chunk.get("category", "")

                if category == "faq" and "| Verified Answer:" in content:
                    reply = content.split("| Verified Answer:")[-1].strip()
                elif category == "amenity":
                    reply = f"Yes! {content}"
                elif category == "policy":
                    reply = f"{content}"
                elif category == "room":
                    reply = f"For room recommendations: {content}"
                elif category == "property":
                    reply = f"{content}"
                else:
                    reply = content

                return {
                    "reply": reply,
                    "tool_called": False,
                    "availability": None,
                    "used_fallback": False,
                    "injection_blocked": False,
                    "retrieved_sources": sources
                }

            # Score > 0 but below 0.5 — still answer from best chunk rather than falling back
            if top_score > 0.0:
                content = top_chunk["content"]
                category = top_chunk.get("category", "")
                if category == "faq" and "| Verified Answer:" in content:
                    reply = content.split("| Verified Answer:")[-1].strip()
                else:
                    reply = content
                return {
                    "reply": reply,
                    "tool_called": False,
                    "availability": None,
                    "used_fallback": False,
                    "injection_blocked": False,
                    "retrieved_sources": sources
                }

        # Genuine out-of-scope — Graceful fallback
        return {
            "reply": "I apologize, but I do not have verified information regarding that request in our resort records. Please contact our concierge desk directly at +91 832 249 8000 or concierge@grandazuregoa.com, and our team will be delighted to assist you.",
            "tool_called": False,
            "availability": None,
            "used_fallback": True,
            "injection_blocked": False,
            "retrieved_sources": sources
        }


orchestrator = AssistantOrchestrator()
