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
  RotateCcw,
  FileCode,
  Copy,
  Check
} from "lucide-react";

import AvailabilityCard, { AvailabilityData } from "./AvailabilityCard";
import DateGuestPicker from "./DateGuestPicker";
import FormattedMessage from "./FormattedMessage";

interface SourcePill {
  title: string;
  category: string;
}

interface LatencyBreakdown {
  hyde_ms?: number;
  search_ms?: number;
  rerank_ms?: number;
  generation_ms?: number;
  total_ms?: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  availability?: AvailabilityData | null;
  needs_dates?: boolean;
  injection_blocked?: boolean;
  used_fallback?: boolean;
  retrieved_sources?: string[];
  /** Detailed citation pills (title + category) from the reranked Top-N */
  sources?: SourcePill[];
  /** Per-stage RAG latency (HyDE / search / rerank / generation ms) */
  latency_breakdown?: LatencyBreakdown | null;
  /** True while tokens are still streaming */
  streaming?: boolean;
}

const BACKEND_URL = (process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000").replace(/\/+$/, "");

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

  // Hotel Data JSON Modal state
  const [showJsonModal, setShowJsonModal] = useState(false);
  const [hotelJsonData, setHotelJsonData] = useState<Record<string, unknown> | null>(null);
  const [isFetchingJson, setIsFetchingJson] = useState(false);
  const [copiedJson, setCopiedJson] = useState(false);
  const [activeJsonTab, setActiveJsonTab] = useState<"all" | "property" | "amenities" | "rooms" | "policies" | "faqs">("all");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
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

    // Append streamed text to the placeholder bubble without re-render churn
    const appendTokens = (text: string) => {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantMsgId ? { ...m, content: m.content + text } : m))
      );
    };
    const finalizeMessage = (meta: Record<string, unknown>, fullReply?: string) => {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? {
                ...m,
                ...(fullReply !== undefined ? { content: fullReply } : {}),
                streaming: false,
                availability: (meta.availability as AvailabilityData | null) ?? null,
                needs_dates: (meta.needs_dates as boolean) ?? false,
                injection_blocked: (meta.injection_blocked as boolean) ?? false,
                used_fallback: (meta.used_fallback as boolean) ?? false,
                retrieved_sources: (meta.retrieved_sources as string[]) ?? [],
                sources: (meta.sources as SourcePill[]) ?? [],
                latency_breakdown: (meta.latency_breakdown as LatencyBreakdown) ?? null
              }
            : m
        )
      );
      if (meta.needs_dates) setShowDatePicker(true);
    };

    try {
      // Phase 6: real-time SSE token streaming from /api/chat/stream.
      const response = await fetch(`${BACKEND_URL}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({
          messages: newHistory.map((m) => ({ role: m.role, content: m.content }))
        }),
        signal: controller.signal
      });

      if (!response.ok || !response.body) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let sseMeta: Record<string, unknown> | null = null;
      let sseWorked = false;

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const evt of events) {
          const line = evt.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          const payload = line.slice(5).trim();
          if (payload === "[DONE]") continue;
          try {
            const data = JSON.parse(payload) as Record<string, unknown>;
            if (data.type === "token" && typeof data.token === "string") {
              sseWorked = true;
              appendTokens(data.token);
            } else if (data.type === "metadata") {
              sseMeta = data;
            } else if (data.type === "error") {
              throw new Error(String(data.message ?? "Streaming error"));
            }
          } catch (parseErr) {
            if (parseErr instanceof Error && parseErr.message !== "Streaming error") continue;
            throw parseErr;
          }
        }
      }

      if (sseMeta) {
        finalizeMessage(sseMeta);
      } else if (sseWorked) {
        // Tokens arrived but no metadata trailer — close the bubble gracefully
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantMsgId ? { ...m, streaming: false } : m))
        );
      } else {
        // SSE yielded nothing usable — fall back to the non-streaming endpoint
        const fallback = await fetch(`${BACKEND_URL}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            messages: newHistory.map((m) => ({ role: m.role, content: m.content }))
          }),
          signal: controller.signal
        });
        if (!fallback.ok) {
          throw new Error(`Server returned ${fallback.status}: ${fallback.statusText}`);
        }
        const data = await fallback.json();
        finalizeMessage(data, data.reply ?? "");
      }
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

  // Open & fetch Hotel Data JSON
  const handleOpenJsonModal = async () => {
    setShowJsonModal(true);
    if (!hotelJsonData && !isFetchingJson) {
      setIsFetchingJson(true);
      try {
        const res = await fetch(`${BACKEND_URL}/api/hotel-data`);
        if (res.ok) {
          const data = await res.json();
          setHotelJsonData(data);
        } else {
          throw new Error("Failed to load hotel data JSON");
        }
      } catch (err) {
        console.error("Failed to load hotel data JSON:", err);
      } finally {
        setIsFetchingJson(false);
      }
    }
  };

  const handleCopyJson = () => {
    if (!hotelJsonData) return;
    const textToCopy =
      activeJsonTab === "all"
        ? JSON.stringify(hotelJsonData, null, 2)
        : JSON.stringify(hotelJsonData[activeJsonTab] ?? {}, null, 2);

    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopiedJson(true);
      setTimeout(() => setCopiedJson(false), 2000);
    });
  };

  // Compute JSON display content based on active tab
  const getDisplayedJson = () => {
    if (!hotelJsonData) return "Loading hotel data...";
    if (activeJsonTab === "all") {
      return JSON.stringify(hotelJsonData, null, 2);
    }
    return JSON.stringify({ [activeJsonTab]: hotelJsonData[activeJsonTab] }, null, 2);
  };

  return (
    <div className="flex flex-col h-full w-full bg-black border-0 sm:border border-[#222222] overflow-hidden shadow-none relative">
      {/* Concierge Console Bar */}
      <div className="px-3 sm:px-4 py-2.5 sm:py-3 bg-black border-b border-[#222222] flex items-center justify-between shrink-0 gap-2">
        {/* Left Branding */}
        <div className="flex items-center gap-2.5 sm:gap-3 min-w-0">
          <div className="w-7 h-7 sm:w-8 sm:h-8 bg-yellow-400 text-black font-mono font-black flex items-center justify-center text-xs tracking-wider shrink-0">
            GA
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5 sm:gap-2">
              <h1 className="text-xs sm:text-sm font-bold tracking-widest uppercase text-white font-mono truncate">
                The Grand Azure
              </h1>
              <span className="hidden min-[420px]:inline text-[9px] sm:text-[10px] text-yellow-400 font-mono tracking-widest px-1 sm:px-1.5 py-0.5 border border-yellow-400/50 bg-[#121212] shrink-0">
                CONCIERGE
              </span>
            </div>
            <p className="text-[10px] sm:text-[11px] text-zinc-400 font-mono truncate hidden sm:block">
              Candolim Beach, Goa • Luxury Beachfront
            </p>
          </div>
        </div>

        {/* Right Header Actions */}
        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
          {/* Hotel Data JSON Button */}
          <button
            type="button"
            onClick={handleOpenJsonModal}
            className="flex items-center gap-1 sm:gap-1.5 px-2 sm:px-2.5 py-1.5 bg-[#111111] hover:bg-yellow-400 hover:text-black text-zinc-300 border border-[#262626] hover:border-yellow-400 text-[10px] sm:text-[11px] font-mono uppercase tracking-wider transition-colors cursor-pointer"
            title="View Hotel Data JSON"
          >
            <FileCode className="w-3.5 h-3.5 text-yellow-400 group-hover:text-black shrink-0" />
            <span className="hidden md:inline">Hotel Data</span>
            <span className="md:hidden">JSON</span>
          </button>

          {/* Quick Property Info Drawer Button */}
          <button
            type="button"
            onClick={() => setShowInfoModal((prev) => !prev)}
            className="flex items-center gap-1 sm:gap-1.5 px-2 sm:px-2.5 py-1.5 bg-[#111111] hover:bg-yellow-400 hover:text-black text-zinc-300 border border-[#262626] hover:border-yellow-400 text-[10px] sm:text-[11px] font-mono uppercase tracking-wider transition-colors cursor-pointer"
            title="View Hotel Policies & Key Facts"
          >
            <Info className="w-3.5 h-3.5 text-yellow-400 group-hover:text-black shrink-0" />
            <span className="hidden md:inline">Hotel Facts</span>
            <span className="md:hidden">Facts</span>
          </button>

          {/* Reset Conversation */}
          <button
            type="button"
            onClick={handleResetChat}
            className="flex items-center gap-1 px-2 sm:px-2.5 py-1.5 bg-[#111111] hover:bg-[#1a1a1a] text-zinc-400 hover:text-white border border-[#262626] text-[10px] sm:text-[11px] font-mono transition-colors cursor-pointer"
            title="Start new conversation"
          >
            <RotateCcw className="w-3 h-3 shrink-0" />
            <span className="hidden lg:inline">Reset</span>
          </button>

          {/* Live Status Indicator */}
          <div className="hidden sm:flex items-center gap-1.5 px-2 sm:px-2.5 py-1.5 bg-[#111111] border border-[#262626] text-yellow-400 text-[10px] sm:text-[11px] font-mono">
            <span className="w-1.5 h-1.5 bg-yellow-400 animate-pulse"></span>
            <span>ONLINE</span>
          </div>
        </div>
      </div>

      {/* Property Facts Drawer */}
      {showInfoModal && (
        <div className="p-3 sm:p-4 bg-[#0a0a0a] border-b border-[#222222] text-xs font-mono space-y-3 shrink-0 max-h-[60vh] sm:max-h-[50vh] overflow-y-auto animate-in fade-in duration-150">
          <div className="flex items-center justify-between border-b border-[#1c1c1c] pb-2">
            <span className="text-yellow-400 uppercase tracking-widest font-bold text-xs flex items-center gap-2">
              <Info className="w-3.5 h-3.5 shrink-0" /> Quick Hotel Guide &amp; Policies
            </span>
            <button
              onClick={() => setShowInfoModal(false)}
              className="text-zinc-400 hover:text-white p-1 cursor-pointer"
              title="Close Guide"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 sm:gap-2.5 text-[11px]">
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5 shrink-0" /> Check-in / Check-out
              </div>
              <div className="text-zinc-300">2:00 PM IST Check-in • 11:00 AM IST Check-out</div>
            </div>
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <Waves className="w-3.5 h-3.5 shrink-0" /> Infinity Pool
              </div>
              <div className="text-zinc-300">Heated oceanview pool open daily 6 AM – 9 PM</div>
            </div>
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <Coffee className="w-3.5 h-3.5 shrink-0" /> Complimentary Breakfast
              </div>
              <div className="text-zinc-300">Royal buffet with Pure Veg &amp; Jain counters (7–10:30 AM)</div>
            </div>
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 shrink-0" /> EV Charging &amp; Valet
              </div>
              <div className="text-zinc-300">Tata Power &amp; universal fast chargers on site</div>
            </div>
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <ShieldCheck className="w-3.5 h-3.5 shrink-0" /> Mandatory ID
              </div>
              <div className="text-zinc-300">Aadhaar / Passport / Voter ID required by law</div>
            </div>
            <div className="p-2.5 bg-black border border-[#222222] space-y-1">
              <div className="text-yellow-400 font-bold flex items-center gap-1.5">
                <Phone className="w-3.5 h-3.5 shrink-0" /> Front Desk
              </div>
              <div className="text-zinc-300">+91 832 249 8000 • concierge@grandazuregoa.com</div>
            </div>
          </div>
        </div>
      )}

      {/* Hotel Data JSON Modal Overlay */}
      {showJsonModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-xs flex items-center justify-center p-2 sm:p-4 md:p-6 animate-in fade-in duration-150">
          <div className="w-full max-w-4xl max-h-[92vh] sm:max-h-[85vh] bg-[#0c0c0c] border border-[#262626] flex flex-col shadow-2xl overflow-hidden">
            {/* Modal Header */}
            <div className="px-3 sm:px-4 py-2.5 sm:py-3 bg-black border-b border-[#222222] flex items-center justify-between shrink-0 gap-2">
              <div className="flex items-center gap-2 min-w-0">
                <FileCode className="w-4 h-4 text-yellow-400 shrink-0" />
                <h3 className="font-mono text-xs sm:text-sm font-bold tracking-wider uppercase text-white truncate">
                  Hotel Ground Truth (<span className="text-yellow-400">hotel_data.json</span>)
                </h3>
              </div>
              <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
                {/* Copy JSON Button */}
                <button
                  type="button"
                  onClick={handleCopyJson}
                  disabled={!hotelJsonData}
                  className="flex items-center gap-1.5 px-2.5 py-1 bg-[#161616] hover:bg-yellow-400 hover:text-black text-zinc-300 border border-[#2e2e2e] hover:border-yellow-400 text-[10px] sm:text-[11px] font-mono uppercase tracking-wider transition-colors cursor-pointer"
                  title="Copy JSON to Clipboard"
                >
                  {copiedJson ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-green-400" />
                      <span className="text-green-400 font-bold">Copied!</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
                {/* Close Button */}
                <button
                  type="button"
                  onClick={() => setShowJsonModal(false)}
                  className="p-1 sm:p-1.5 text-zinc-400 hover:text-white hover:bg-[#1a1a1a] transition-colors cursor-pointer"
                  title="Close JSON Viewer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Filter Tabs */}
            <div className="px-3 sm:px-4 py-1.5 bg-[#101010] border-b border-[#1c1c1c] flex items-center gap-1 sm:gap-1.5 overflow-x-auto no-scrollbar shrink-0">
              <span className="text-[10px] font-mono uppercase tracking-wider text-zinc-500 mr-1 hidden sm:inline">
                View:
              </span>
              {[
                { id: "all", label: "Full JSON" },
                { id: "property", label: "Property" },
                { id: "amenities", label: "Amenities" },
                { id: "rooms", label: "Rooms" },
                { id: "policies", label: "Policies" },
                { id: "faqs", label: "FAQs" }
              ].map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  onClick={() => setActiveJsonTab(tab.id as typeof activeJsonTab)}
                  className={`px-2 py-1 text-[10px] sm:text-[11px] font-mono uppercase tracking-wider transition-colors cursor-pointer shrink-0 ${
                    activeJsonTab === tab.id
                      ? "bg-yellow-400 text-black font-bold"
                      : "bg-[#161616] text-zinc-400 hover:text-white border border-[#242424]"
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* JSON Code Viewer */}
            <div className="flex-1 overflow-auto p-3 sm:p-4 bg-black font-mono text-[11px] sm:text-xs leading-relaxed text-zinc-300 select-text">
              {isFetchingJson ? (
                <div className="flex items-center justify-center py-12 gap-2 text-zinc-400">
                  <RefreshCw className="w-4 h-4 animate-spin text-yellow-400" />
                  <span>Loading hotel_data.json...</span>
                </div>
              ) : (
                <pre className="whitespace-pre overflow-x-auto scrollbar-thin">
                  <code>{getDisplayedJson()}</code>
                </pre>
              )}
            </div>

            {/* Modal Footer */}
            <div className="px-3 sm:px-4 py-2 bg-[#080808] border-t border-[#1c1c1c] flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-zinc-500 shrink-0">
              <span>Ground Truth Dataset: Verified Knowledge Base (Property, Amenities, Rooms, Policies, FAQs)</span>
              <span>Candolim Beach, North Goa 403515</span>
            </div>
          </div>
        </div>
      )}

      {/* Chat Messages Feed */}
      <div className="flex-1 overflow-y-auto p-3 sm:p-4 md:p-5 space-y-3.5 sm:space-y-4 bg-black">
        {messages.map((m) => {
          const isUser = m.role === "user";
          return (
            <div
              key={m.id}
              className={`flex gap-2.5 sm:gap-3 max-w-[95%] sm:max-w-[88%] md:max-w-[82%] ${
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

              <div className="flex flex-col space-y-1 min-w-0">
                <div
                  className={`p-3 sm:p-3.5 text-xs sm:text-sm leading-relaxed ${
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

                  {((m.sources && m.sources.length > 0) ||
                    (m.retrieved_sources && m.retrieved_sources.length > 0)) &&
                    !isUser &&
                    !m.streaming && (
                      <div className="mt-3 pt-2.5 border-t border-[#1e1e1e] flex flex-wrap items-center gap-1.5 text-[10px] font-mono text-zinc-400">
                        <Compass className="w-3 h-3 text-yellow-400 shrink-0" />
                        <span className="text-zinc-500 uppercase">Knowledge Base:</span>
                        {(m.sources && m.sources.length > 0
                          ? m.sources.slice(0, 3).map((s) => s.title)
                          : (m.retrieved_sources ?? []).slice(0, 3)
                        ).map((src, i) => (
                          <span
                            key={i}
                            className="bg-[#141414] text-yellow-400 px-1.5 py-0.5 border border-[#262626]"
                          >
                            {src}
                          </span>
                        ))}
                        {m.latency_breakdown && (
                          <span className="text-zinc-600">
                            {[
                              m.latency_breakdown.search_ms !== undefined &&
                                `search ${Math.round(m.latency_breakdown.search_ms)}ms`,
                              m.latency_breakdown.rerank_ms !== undefined &&
                                `rerank ${Math.round(m.latency_breakdown.rerank_ms)}ms`,
                              m.latency_breakdown.generation_ms !== undefined &&
                                `gen ${Math.round(m.latency_breakdown.generation_ms)}ms`
                            ]
                              .filter(Boolean)
                              .join(" • ")}
                          </span>
                        )}
                      </div>
                    )}
                </div>
              </div>
            </div>
          );
        })}

        {errorMessage && (
          <div className="p-2.5 sm:p-3 bg-[#1c1114] border border-red-500/50 text-red-300 text-xs font-mono flex items-center justify-between gap-2">
            <span className="truncate">{errorMessage}</span>
            <button
              onClick={() => handleSendMessage()}
              className="flex items-center gap-1 px-2.5 py-1 bg-red-950/80 hover:bg-red-900 text-red-200 border border-red-500/40 text-[11px] font-bold transition-colors cursor-pointer shrink-0"
            >
              <RefreshCw className="w-3 h-3" /> Retry
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {showDatePicker && (
        <div className="p-2.5 sm:p-3 border-t border-[#222222] bg-black">
          <DateGuestPicker onSearch={handleFormSearch} isLoading={isLoading} />
        </div>
      )}

      {/* Suggested Quick Prompts */}
      <div className="px-2.5 sm:px-3 py-2 border-t border-[#1c1c1c] bg-black">
        <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar py-0.5">
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
              className="text-[10px] sm:text-[11px] font-mono shrink-0 px-2 sm:px-2.5 py-1 bg-[#111111] hover:bg-yellow-400 hover:text-black text-zinc-300 border border-[#242424] hover:border-yellow-400 transition-colors cursor-pointer"
            >
              {prompt}
            </button>
          ))}
        </div>
      </div>

      {/* Message Input Box */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSendMessage();
        }}
        className="p-2.5 sm:p-3 border-t border-[#222222] bg-black flex items-center gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question or inquire about rooms..."
          disabled={isLoading}
          className="flex-1 px-3 sm:px-3.5 py-2 sm:py-2.5 bg-black border border-[#282828] focus:border-yellow-400 text-white placeholder-zinc-500 text-xs sm:text-sm outline-none transition-colors font-sans"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="px-3 sm:px-4 py-2 sm:py-2.5 bg-yellow-400 hover:bg-yellow-300 disabled:bg-[#1a1a1a] text-black disabled:text-zinc-600 font-bold transition-colors cursor-pointer shrink-0 font-mono text-xs uppercase tracking-wider flex items-center gap-1.5"
        >
          <span className="hidden sm:inline">Send</span>
          <Send className="w-3.5 h-3.5" />
        </button>
      </form>
    </div>
  );
}
