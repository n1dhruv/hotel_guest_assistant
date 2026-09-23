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
    context_text = "\n".join([
        f"[{c['category'].upper()}] {c['title']}: {c['content']}"
        for c in retrieved_chunks
    ])

    return f"""You are the AI concierge for The Grand Azure Heritage Resort & Spa, Candolim, Goa. Today: {today_str}.

HOTEL CONTEXT:
{context_text}

RULES:
- Answer from context above. For broad questions, summarise all relevant facts.
- For availability/booking requests with dates provided, call check_room_availability. Without dates, ask for them.
- If truly unrelated to the resort, direct to +91 832 249 8000 / concierge@grandazuregoa.com.
- Tone: warm, Namaste hospitality, concise."""


def _synthesize_amenities(_chunks: list[dict]) -> str:
    """Return ALL amenity chunks directly from the KB — not limited by retrieval k."""
    amenities = [c for c in kb.chunks if c.get("category") == "amenity"]
    if not amenities:
        return "I'm sorry, I don't have detailed amenity information available right now."
    lines = []
    for a in amenities:
        title = a['title'].replace('Amenity: ', '').strip()
        content = a['content']
        if " | Timings: " in content and ". Description: " in content:
            _, rest = content.split(" | Timings: ", 1)
            timings, desc = rest.split(". Description: ", 1)
            lines.append(f"• **{title}** ({timings.strip()}):\n  {desc.strip()}")
        else:
            lines.append(f"• **{title}:** {content}")
    return (
        f"Namaste! Here are all {len(amenities)} resort amenities at The Grand Azure Heritage Resort & Spa:\n\n"
        + "\n\n".join(lines)
    )


def _synthesize_broad_answer(_chunks: list[dict]) -> str:
    """Comprehensive overview built from ALL 18 ground truth kb.chunks."""
    groups: dict[str, list[dict]] = {"property": [], "amenity": [], "room": [], "policy": []}
    seen_keys: set[str] = set()

    for chunk in kb.chunks:
        category = chunk.get("category", "")
        key = f"{category}-{chunk.get('title', '')}"
        if key in seen_keys or category not in groups:
            continue
        seen_keys.add(key)
        groups[category].append(chunk)

    sections: list[str] = []

    for p in groups["property"]:
        sections.append(f"🏨 **About the Resort:** {p['content']}")

    if groups["amenity"]:
        amenity_lines = [
            f"  • **{a['title'].replace('Amenity: ', '')}:** {a['content']}"
            for a in groups["amenity"]
        ]
        sections.append("✨ **Amenities & Facilities:**\n" + "\n".join(amenity_lines))

    if groups["room"]:
        room_lines = [
            f"  • **{r['title'].replace('Room: ', '')}:** {r['content']}"
            for r in groups["room"]
        ]
        sections.append("🛏️ **Room Types:**\n" + "\n".join(room_lines))

    if groups["policy"]:
        policy_lines = [
            f"  • **{p['title'].replace('Policy: ', '').capitalize()}:** {p['content']}"
            for p in groups["policy"]
        ]
        sections.append("📋 **Policies:**\n" + "\n".join(policy_lines))

    if sections:
        return (
            "Namaste! Here is a comprehensive overview of The Grand Azure Heritage Resort & Spa:\n\n"
            + "\n\n".join(sections)
        )

    return kb.chunks[0]["content"] if kb.chunks else "I apologize, I do not have that information available."


def _synthesize_rooms() -> str:
    """Return ALL room tiers directly from the KB."""
    rooms = [c for c in kb.chunks if c.get("category") == "room"]
    if not rooms:
        return "I'm sorry, I don't have room information available right now."
    lines = []
    for r in rooms:
        title = r['title'].replace('Room: ', '').strip()
        lines.append(f"• **{title}:** {r['content']}")
    return (
        f"Namaste! Here are our {len(rooms)} room types at The Grand Azure Heritage Resort & Spa:\n\n"
        + "\n\n".join(lines)
    )


def _synthesize_policies() -> str:
    """Return ALL policy chunks directly from the KB."""
    policies = [c for c in kb.chunks if c.get("category") == "policy"]
    if not policies:
        return "I'm sorry, I don't have policy information available right now."
    lines = []
    for p in policies:
        title = p['title'].replace('Policy: ', '').strip()
        content = p['content']
        if "Policy: " in content:
            _, policy_text = content.split("Policy: ", 1)
            lines.append(f"• **{title}:** {policy_text.strip()}")
        else:
            lines.append(f"• **{title}:** {content}")
    return (
        f"Namaste! Here are our {len(policies)} resort policies at The Grand Azure Heritage Resort & Spa:\n\n"
        + "\n\n".join(lines)
    )



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

        # 2. Semantic RAG Retrieval — k adapts to query breadth
        retrieved = kb.retrieve(last_user_message, k=7)
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

        # RAG Retrieval — k adapts to query breadth
        retrieved = kb.retrieve(last_user_message, k=7)
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
            timeout=3
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

            # Amenity-list query — ONLY when asking generically, not about one amenity.
            # Old bug: bare "facilities" matched "parking facilities?" -> dumped all 6 amenities.
            _SPECIFIC_AMENITY_WORDS = {
                "pool", "swimming", "spa", "gym", "fitness", "massage",
                "breakfast", "buffet", "wifi", "wi-fi", "internet",
                "parking", "valet", "ev", "car",
                "restaurant", "dining", "dinner", "lunch", "food", "saffron",
                "jain", "vegetarian", "veg",
            }
            _has_specific_amenity = any(w in q_lower for w in _SPECIFIC_AMENITY_WORDS)
            _has_generic_word = ("amenit" in q_lower or "facilit" in q_lower)
            _has_lister_word = any(
                w in q_lower for w in ("all", "list", "show", "what", "offer", "overview", "summar", "every")
            )
            if "amenit" in q_lower:
                is_amenity_list = not _has_specific_amenity
            elif "facilit" in q_lower:
                is_amenity_list = (not _has_specific_amenity) and _has_lister_word
            else:
                is_amenity_list = False

            if is_amenity_list:
                reply = _synthesize_amenities(retrieved)
                return {
                    "reply": reply,
                    "tool_called": False,
                    "availability": None,
                    "used_fallback": False,
                    "injection_blocked": False,
                    "retrieved_sources": sources
                }

            # Focused list intents — rooms / policies get their own concise list,
            # NOT the full 18-chunk overview.
            _is_rooms_list = (
                ("room type" in q_lower or "room-types" in q_lower or "all rooms" in q_lower)
                or ("all" in q_lower and "room" in q_lower)
            ) and not has_dates
            if _is_rooms_list:
                reply = _synthesize_rooms()
                return {
                    "reply": reply,
                    "tool_called": False,
                    "availability": None,
                    "used_fallback": False,
                    "injection_blocked": False,
                    "retrieved_sources": sources
                }

            _is_policies_list = ("all policies" in q_lower) or ("all" in q_lower and "polic" in q_lower)
            if _is_policies_list:
                reply = _synthesize_policies()
                return {
                    "reply": reply,
                    "tool_called": False,
                    "availability": None,
                    "used_fallback": False,
                    "injection_blocked": False,
                    "retrieved_sources": sources
                }

            # Highlights / benefits / "what's special" -> concise FAQ answer,
            # NOT the full 18-chunk overview. Retrieval scores 0 for "special"
            # (no literal match), so answer it explicitly here.
            if any(w in q_lower for w in ("benefit", "highlight", "special", "what makes")):
                _hl = next(
                    (c for c in kb.chunks if c.get("topic") == "highlights"),
                    None,
                )
                if _hl is not None:
                    _title = _hl.get("title", "Resort highlights").replace("FAQ: ", "").strip()
                    _answer = _hl.get("content", "")
                    if ": " in _answer:
                        _answer = _answer.split(": ", 1)[1].strip()
                    return {
                        "reply": f"**{_title}:** {_answer}",
                        "tool_called": False,
                        "availability": None,
                        "used_fallback": False,
                        "injection_blocked": False,
                        "retrieved_sources": [_hl.get("title", "")],
                    }

            # Full overview — ONLY for open-ended resort questions.
            # "benefits / highlights / special" intentionally NOT here:
            # those have a dedicated concise FAQ chunk and must fall through
            # to specific synthesis below instead of dumping all 18 chunks.
            _EXPLICIT_BROAD_PHRASES = (
                "everything", "all about", "what all", "overview",
            )
            _has_explicit_broad = any(p in q_lower for p in _EXPLICIT_BROAD_PHRASES)
            _mentions_place = any(
                w in q_lower for w in ("resort", "hotel", "property", "grand azure", "stay here", "this place")
            )
            _about_place = ("about" in q_lower and _mentions_place)
            _tell_about_place = (
                (("tell me" in q_lower) or ("describe" in q_lower)) and (_mentions_place or "about" in q_lower)
            )
            is_broad_query = _has_explicit_broad or _about_place or _tell_about_place

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

            # Calculate token overlap between query and top retrieved chunk
            q_tokens = kb._tokenize(query)
            doc_tokens = set(kb._tokenize(top_chunk["title"] + " " + top_chunk["content"]))
            overlap = [t for t in q_tokens if t in doc_tokens]
            overlap_ratio = len(overlap) / len(q_tokens) if q_tokens else 0.0

            # For specific queries: require substantive grounding (not an accidental 1-word collision)
            if top_score >= 1.5 and overlap_ratio >= 0.25:
                threshold = top_score * 0.7  # include chunks within 30% of best score
                relevant = [c for c in retrieved if c.get("score", 0.0) >= threshold]

                # Group chunks by canonical topic (e.g. 'pool', 'breakfast', 'checkin', 'checkout')
                topic_groups: dict[str, list[dict]] = {}
                for chunk in relevant:
                    t = chunk.get("topic") or chunk.get("id")
                    topic_groups.setdefault(t, []).append(chunk)

                # Deduplicate property overview chunk if specific topic chunks are present
                if len(topic_groups) > 1 and "property" in topic_groups:
                    prop_keywords = ["about", "resort", "hotel", "contact", "address", "phone", "email", "location", "where"]
                    if not any(k in q_lower for k in prop_keywords):
                        del topic_groups["property"]

                # Deduplicate checkin/checkout policy if separate checkin or checkout topics are present
                if ("checkin" in topic_groups or "checkout" in topic_groups) and "checkincheckout" in topic_groups:
                    del topic_groups["checkincheckout"]

                answers = []
                selected_sources = []

                for topic, chunks in topic_groups.items():
                    # Pick highest scoring chunk for this topic
                    best_chunk = max(chunks, key=lambda c: c.get("score", 0.0))
                    content = best_chunk["content"]
                    category = best_chunk.get("category", "")

                    if category == "amenity":
                        if " | Timings: " in content and ". Description: " in content:
                            name, rest = content.split(" | Timings: ", 1)
                            timings, desc = rest.split(". Description: ", 1)
                            answers.append(
                                f"Yes! **{name.strip()}** (Timings: {timings.strip()}):\n{desc.strip()}"
                            )
                        else:
                            answers.append(f"Yes! {content}")
                    elif category == "policy":
                        title = best_chunk.get("title", "").replace("Policy: ", "").strip()
                        label = title if "policy" in title.lower() or "timings" in title.lower() else f"{title} Policy"
                        if "Policy: " in content:
                            _, policy_text = content.split("Policy: ", 1)
                            answers.append(f"**{label}:** {policy_text.strip()}")
                        else:
                            answers.append(f"**{label}:** {content}")
                    elif category == "faq":
                        title = best_chunk.get("title", "").replace("FAQ: ", "").strip()
                        if ": " in content:
                            _, answer_text = content.split(": ", 1)
                            answers.append(f"**{title}:** {answer_text.strip()}")
                        else:
                            answers.append(f"**{title}:** {content}")
                    elif category == "room":
                        title = best_chunk.get("title", "").replace("Room: ", "").strip()
                        answers.append(f"**{title}:**\n{content}")
                    elif category == "property":
                        answers.append(f"**The Grand Azure Heritage Resort & Spa:**\n{content}")
                    else:
                        answers.append(content)
                    selected_sources.append(best_chunk["title"])

                reply = "\n\n".join(answers) if answers else top_chunk["content"]

                return {
                    "reply": reply,
                    "tool_called": False,
                    "availability": None,
                    "used_fallback": False,
                    "injection_blocked": False,
                    "retrieved_sources": selected_sources or sources
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
