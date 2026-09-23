"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Bot,
  User,
  ShieldAlert,
  Sparkles,
  RefreshCw,
  Compass,
  Info,
  X,
  Clock,
  Waves,
  Coffee,
  Zap,
  ShieldCheck,
  Phone,
  RotateCcw
} from "lucide-react";

import AvailabilityCard, { AvailabilityData } from "./AvailabilityCard";
import DateGuestPicker from "./DateGuestPicker";
import FormattedMessage from "./FormattedMessage";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  availability?: AvailabilityData | null;
  needs_dates?: boolean;
  injection_blocked?: boolean;
  used_fallback?: boolean;
  retrieved_sources?: string[];
  /** True while the typewriter animation is still running */
  streaming?: boolean;
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const INITIAL_MESSAGE: Message = {
  id: "welcome-msg",
  role: "assistant",
  content:
    "Welcome to The Grand Azure Resort & Spa, Candolim, Goa. I am your AI Stay Concierge. How may I assist you today? You can query room availability, tariffs, resort amenities, dining, or property policies."
};

const QUICK_PROMPTS = [
  "Check room availability",
  "What time is check-in & check-out?",
  "Does the resort have an infinity pool?",
  "What is the cancellation policy?",
  "Is breakfast complimentary?",
  "What government ID is required?"
];

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([INITIAL_MESSAGE]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [showInfoModal, setShowInfoModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  // Keeps the current typewriter interval so we can cancel on abort/reset
  const typewriterRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, showDatePicker]);

  const handleSendMessage = async (textToSend?: string) => {
    const text = (textToSend || input).trim();
    if (!text || isLoading) return;

    setInput("");
    setErrorMessage(null);

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content: text
    };

    const newHistory = [...messages, userMessage];
    setMessages(newHistory);
    setIsLoading(true);

    const assistantMsgId = `assistant-${Date.now()}`;

    // Add placeholder with blinking cursor while the fetch is in-flight
    setMessages((prev) => [
      ...prev,
      { id: assistantMsgId, role: "assistant", content: "", streaming: true } as Message
    ]);

    // Abort any in-flight request
    if (abortControllerRef.current) abortControllerRef.current.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      // ── Single regular fetch — no SSE overhead ─────────────────────────────
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: newHistory.map((m) => ({ role: m.role, content: m.content }))
        }),
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      const fullReply: string = data.reply ?? "";

      // ── Typewriter animation — words revealed progressively ────────────────
      // The network round-trip is already done; we're just animating the display.
      const words = fullReply.split(" ");
      let wordIdx = 0;

      await new Promise<void>((resolve) => {
        const tick = () => {
          // Reveal 3 words per tick → ~100 wpm at 30 ms/tick, looks natural
          const batch = words.slice(wordIdx, wordIdx + 3).join(" ");
          const isFirst = wordIdx === 0;
          wordIdx += 3;

          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: m.content + (isFirst ? "" : " ") + batch }
                : m
            )
          );

          if (wordIdx < words.length) {
            typewriterRef.current = setTimeout(tick, 30);
          } else {
            // Final commit: set exact full text + all metadata
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantMsgId
                  ? {
                      ...m,
                      content: fullReply,
                      streaming: false,
                      availability: data.availability ?? null,
                      needs_dates: data.needs_dates ?? false,
                      injection_blocked: data.injection_blocked ?? false,
                      used_fallback: data.used_fallback ?? false,
                      retrieved_sources: data.retrieved_sources ?? []
                    }
                  : m
              )
            );
            if (data.needs_dates) setShowDatePicker(true);
            resolve();
          }
        };

        tick(); // start immediately
      });
    } catch (err: unknown) {
      if (typewriterRef.current) clearTimeout(typewriterRef.current);
      if (err instanceof Error && err.name === "AbortError") {
        setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId));
      } else {
        console.error("Chat error:", err);
        setMessages((prev) => prev.filter((m) => m.id !== assistantMsgId));
        setErrorMessage(
          "Could not connect to resort assistant backend. Please ensure the server is running on port 8000."
        );
      }
    } finally {
      // Ensure streaming flag cleared even if something went wrong mid-animation
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantMsgId ? { ...m, streaming: false } : m))
      );
      setIsLoading(false);
    }
  };

  const handleFormSearch = (checkIn: string, checkOut: string, adults: number) => {
    setShowDatePicker(false);
    handleSendMessage(`Are there rooms available from ${checkIn} to ${checkOut} for ${adults} adults?`);
  };

  const handleResetChat = () => {
    if (abortControllerRef.current) abortControllerRef.current.abort();
    if (typewriterRef.current) clearTimeout(typewriterRef.current);
    setMessages([INITIAL_MESSAGE]);
    setShowDatePicker(false);
    setErrorMessage(null);
  };

  return (
    <div className="flex flex-col h-full w-full bg-black border border-[#222222] overflow-hidden shadow-none">
      {/* Concierge Console Bar */}
      <div className="px-4 py-3 bg-black border-b border-[#222222] flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-yellow-400 text-black font-mono font-black flex items-center justify-center text-xs tracking-wider">
            GA
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xs sm:text-sm font-bold tracking-widest uppercase text-white font-mono">
                The Grand Azure
              </h1>
              <span className="text-[10px] text-yellow-400 font-mono tracking-widest px-1.5 py-0.5 border border-yellow-400/50 bg-[#121212]">
                AI CONCIERGE
              </span>
            </div>
            <p className="text-[11px] text-zinc-400 font-mono">
              Candolim Beach, Goa • Luxury Beachfront
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowInfoModal((prev) => !prev)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 bg-[#111111] hover:bg-yellow-400 hover:text-black text-zinc-300 border border-[#262626] hover:border-yellow-400 text-[11px] font-mono uppercase tracking-wider transition-colors cursor-pointer"
            title="View Hotel Policies & Key Facts"
          >
            <Info className="w-3.5 h-3.5 text-yellow-400 group-hover:text-black" />
            <span className="hidden sm:inline">Hotel Facts</span>
          </button>

          <button
            type="button"
            onClick={handleResetChat}
            className="flex items-center gap-1 px-2.5 py-1.5 bg-[#111111] hover:bg-[#1a1a1a] text-zinc-400 hover:text-white border border-[#262626] text-[11px] font-mono transition-colors cursor-pointer"
            title="Start new conversation"
          >
            <RotateCcw className="w-3 h-3" />
            <span className="hidden md:inline">Reset</span>
          </button>

          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 bg-[#111111] border border-[#262626] text-yellow-400 text-[11px] font-mono">
            <span className="w-1.5 h-1.5 bg-yellow-400"></span>
            <span>ONLINE</span>
          </div>
        </div>
      </div>

      {/* Property Facts Modal */}
      {showInfoModal && (
        <div className="p-4 bg-[#0a0a0a] border-b border-[#222222] text-xs font-mono space-y-3 shrink-0 animate-in fade-in duration-150">
          <div className="flex items-center justify-between border-b border-[#1c1c1c] pb-2">
            <span className="text-yellow-400 uppercase tracking-widest font-bold text-xs flex items-center gap-2">
              <Info className="w-3.5 h-3.5" /> Quick Hotel Guide &amp; Policies
            </span>
            <button
              onClick={() => setShowInfoModal(false)}
              className="text-zinc-400 hover:text-white p-1 cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5 text-[11px]">
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5" /> Check-in / Check-out
              </div>
              <div className="text-zinc-300">2:00 PM IST Check-in • 11:00 AM IST Check-out</div>
            </div>
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <Waves className="w-3.5 h-3.5" /> Infinity Pool
              </div>
              <div className="text-zinc-300">Heated oceanview pool open daily 6 AM – 9 PM</div>
            </div>
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <Coffee className="w-3.5 h-3.5" /> Complimentary Breakfast
              </div>
              <div className="text-zinc-300">Royal buffet with Pure Veg &amp; Jain counters (7–10:30 AM)</div>
            </div>
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5" /> EV Charging &amp; Valet
              </div>
              <div className="text-zinc-300">Tata Power &amp; universal fast chargers on site</div>
            </div>
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5" /> Mandatory ID
              </div>
              <div className="text-zinc-300">Aadhaar / Passport / Voter ID required by law</div>
            </div>
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <Phone className="w-3.5 h-3.5" /> Front Desk
              </div>
              <div className="text-zinc-300">+91 832 249 8000 • concierge@grandazuregoa.com</div>
            </div>
          </div>
        </div>
      )}

      {/* Chat Messages Feed */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4 bg-black">
        {messages.map((m) => {
          const isUser = m.role === "user";
          return (
            <div
              key={m.id}
              className={`flex gap-3 max-w-[92%] sm:max-w-[85%] ${
                isUser ? "ml-auto flex-row-reverse" : "mr-auto"
              }`}
            >
              <div
                className={`w-7 h-7 flex items-center justify-center shrink-0 text-xs font-mono font-bold ${
                  isUser
                    ? "bg-yellow-400 text-black"
                    : "bg-[#141414] text-yellow-400 border border-[#262626]"
                }`}
              >
                {isUser ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
              </div>

              <div className="flex flex-col space-y-1">
                <div
                  className={`p-3.5 text-xs sm:text-sm leading-relaxed ${
                    isUser
                      ? "bg-[#141414] text-white border-l-2 border-yellow-400"
                      : "bg-[#0c0c0c] text-zinc-100 border border-[#222222]"
                  }`}
                >
                  {m.injection_blocked && (
                    <div className="flex items-center gap-1.5 mb-2.5 text-red-300 font-mono text-xs bg-[#241215] border border-red-500/50 p-2">
                      <ShieldAlert className="w-4 h-4 text-red-400 shrink-0" />
                      <span>Security Guard Active: Input flagged or prompt injection mitigated.</span>
                    </div>
                  )}

                  {/* Empty state: show pulsing dots while waiting for first word */}
                  {m.streaming && m.content === "" ? (
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-mono text-zinc-400">Thinking</span>
                      <span className="w-1.5 h-1.5 bg-yellow-400 rounded-full animate-bounce [animation-delay:0ms]" />
                      <span className="w-1.5 h-1.5 bg-yellow-400 rounded-full animate-bounce [animation-delay:150ms]" />
                      <span className="w-1.5 h-1.5 bg-yellow-400 rounded-full animate-bounce [animation-delay:300ms]" />
                    </div>
                  ) : (
                    <>
                      <FormattedMessage content={m.content} isUser={isUser} />
                      {/* Blinking cursor while typewriter is running */}
                      {m.streaming && (
                        <span className="inline-block w-2 h-[1em] bg-yellow-400 animate-pulse ml-0.5 align-middle" />
                      )}
                    </>
                  )}

                  {!m.streaming && m.availability && <AvailabilityCard data={m.availability} />}

                  {!m.streaming && m.needs_dates && (
                    <div className="mt-3">
                      <DateGuestPicker onSearch={handleFormSearch} isLoading={isLoading} />
                    </div>
                  )}

                  {m.retrieved_sources && m.retrieved_sources.length > 0 && !isUser && !m.streaming && (
                    <div className="mt-3 pt-2.5 border-t border-[#1e1e1e] flex flex-wrap items-center gap-1.5 text-[10px] font-mono text-zinc-400">
                      <Compass className="w-3 h-3 text-yellow-400" />
                      <span className="text-zinc-500 uppercase">Knowledge Base:</span>
                      {m.retrieved_sources.slice(0, 2).map((src, i) => (
                        <span
                          key={i}
                          className="bg-[#141414] text-yellow-400 px-1.5 py-0.5 border border-[#262626]"
                        >
                          {src}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}

        {errorMessage && (
          <div className="p-3 bg-[#1c1114] border border-red-500/50 text-red-300 text-xs font-mono flex items-center justify-between">
            <span>{errorMessage}</span>
            <button
              onClick={() => handleSendMessage()}
              className="flex items-center gap-1 px-2.5 py-1 bg-red-950/80 hover:bg-red-900 text-red-200 border border-red-500/40 text-[11px] font-bold transition-colors cursor-pointer"
            >
              <RefreshCw className="w-3 h-3" /> Retry
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {showDatePicker && (
        <div className="p-3 border-t border-[#222222] bg-black">
          <DateGuestPicker onSearch={handleFormSearch} isLoading={isLoading} />
        </div>
      )}

      {/* Quick Prompts */}
      <div className="px-3 py-2 border-t border-[#1c1c1c] bg-black">
        <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar py-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-yellow-400 shrink-0 flex items-center gap-1 mr-1">
            <Sparkles className="w-3 h-3 text-yellow-400" /> Prompts:
          </span>
          {QUICK_PROMPTS.map((prompt, idx) => (
            <button
              key={idx}
              type="button"
              disabled={isLoading}
              onClick={() => {
                if (prompt === "Check room availability") {
                  setShowDatePicker((prev) => !prev);
                } else {
                  handleSendMessage(prompt);
                }
              }}
              className="text-[11px] font-mono shrink-0 px-2.5 py-1 bg-[#111111] hover:bg-yellow-400 hover:text-black text-zinc-300 border border-[#242424] hover:border-yellow-400 transition-colors cursor-pointer"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Input Box */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSendMessage();
        }}
        className="p-3 border-t border-[#222222] bg-black flex items-center gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question or inquire about rooms..."
          disabled={isLoading}
          className="flex-1 px-3.5 py-2.5 bg-black border border-[#282828] focus:border-yellow-400 text-white placeholder-zinc-500 text-xs sm:text-sm outline-none transition-colors font-sans"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="px-4 py-2.5 bg-yellow-400 hover:bg-yellow-300 disabled:bg-[#1a1a1a] text-black disabled:text-zinc-600 font-bold transition-colors cursor-pointer shrink-0 font-mono text-xs uppercase tracking-wider flex items-center gap-1.5"
        >
          <span className="hidden sm:inline">Send</span>
          <Send className="w-3.5 h-3.5" />
        </button>
      </form>
    </div>
  );
}
