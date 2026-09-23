"use client";

import React, { useState, useRef, useEffect } from "react";
import { Send, Bot, User, ShieldAlert, Sparkles, RefreshCw, Compass } from "lucide-react";
import AvailabilityCard, { AvailabilityData } from "./AvailabilityCard";
import DateGuestPicker from "./DateGuestPicker";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  availability?: AvailabilityData | null;
  needs_dates?: boolean;
  injection_blocked?: boolean;
  used_fallback?: boolean;
  retrieved_sources?: string[];
}

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

const QUICK_PROMPTS = [
  "What time is check-in & check-out?",
  "Does the resort have an infinity pool?",
  "Check room availability",
  "What is the cancellation policy?",
  "Is breakfast included?",
  "What government ID is required?"
];

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome-msg",
      role: "assistant",
      content:
        "Namaste and welcome to The Grand Azure Heritage Resort & Spa, Candolim, Goa! 🌴 I am your AI Guest Concierge. How may I assist you with your upcoming stay? You can ask about our rooms, dining, pool, policies, or check live room availability and tariffs."
    }
  ]);

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [showDatePicker, setShowDatePicker] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

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

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: newHistory.map((m) => ({ role: m.role, content: m.content }))
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.reply,
        availability: data.availability,
        needs_dates: data.needs_dates,
        injection_blocked: data.injection_blocked,
        used_fallback: data.used_fallback,
        retrieved_sources: data.retrieved_sources
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (data.needs_dates) {
        setShowDatePicker(true);
      }
    } catch (err: any) {
      console.error("Chat error:", err);
      setErrorMessage("Could not connect to the resort assistant service. Please verify that the backend server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFormSearch = (checkIn: string, checkOut: string, adults: number) => {
    setShowDatePicker(false);
    handleSendMessage(`Are there rooms available from ${checkIn} to ${checkOut} for ${adults} adults?`);
  };

  return (
    <div className="flex flex-col h-[740px] max-h-[85vh] bg-white rounded-2xl shadow-xl border border-slate-200/90 overflow-hidden">
      {/* Chat Messages Feed */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-5 space-y-4 bg-slate-50/50">
        {messages.map((m) => {
          const isUser = m.role === "user";

          return (
            <div
              key={m.id}
              className={`flex gap-3 max-w-[90%] sm:max-w-[82%] ${isUser ? "ml-auto flex-row-reverse" : "mr-auto"}`}
            >
              {/* Avatar Icon */}
              <div
                className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 text-xs font-semibold shadow-xs ${
                  isUser
                    ? "bg-slate-800 text-white"
                    : "bg-emerald-700 text-amber-200 ring-2 ring-emerald-600/20"
                }`}
              >
                {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
              </div>

              {/* Message Content Bubble */}
              <div className="flex flex-col space-y-1">
                <div
                  className={`p-3.5 rounded-2xl text-xs sm:text-sm leading-relaxed shadow-xs ${
                    isUser
                      ? "bg-slate-900 text-white rounded-tr-none"
                      : "bg-white text-slate-800 border border-slate-200 rounded-tl-none"
                  }`}
                >
                  {m.injection_blocked && (
                    <div className="flex items-center gap-1.5 mb-2 text-rose-600 font-semibold text-xs bg-rose-50 border border-rose-200 p-2 rounded-lg">
                      <ShieldAlert className="w-4 h-4 shrink-0" />
                      <span>Security Guard Active: Jailbreak / Prompt Injection Prevented</span>
                    </div>
                  )}

                  <p className="whitespace-pre-wrap">{m.content}</p>

                  {/* Structured Availability Cards (if tool called) */}
                  {m.availability && <AvailabilityCard data={m.availability} />}

                  {/* Inline Date Selector if assistant prompted for missing dates */}
                  {m.needs_dates && (
                    <div className="mt-3">
                      <DateGuestPicker onSearch={handleFormSearch} isLoading={isLoading} />
                    </div>
                  )}

                  {/* Verified Sources Badge (RAG-Lite evidence) */}
                  {m.retrieved_sources && m.retrieved_sources.length > 0 && !isUser && (
                    <div className="mt-2.5 pt-2 border-t border-slate-100 flex flex-wrap items-center gap-1 text-[11px] text-slate-400">
                      <Compass className="w-3 h-3 text-emerald-600" />
                      <span className="font-medium text-slate-500">Verified Knowledge:</span>
                      {m.retrieved_sources.slice(0, 2).map((src, i) => (
                        <span key={i} className="bg-slate-100 text-slate-600 px-1.5 py-0.5 rounded text-[10px]">
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

        {/* Loading Pulse */}
        {isLoading && (
          <div className="flex gap-3 max-w-[80%] mr-auto items-center">
            <div className="w-8 h-8 rounded-full bg-emerald-700 text-amber-200 flex items-center justify-center shrink-0">
              <Bot className="w-4 h-4 animate-spin" />
            </div>
            <div className="p-3 bg-white border border-slate-200 rounded-2xl rounded-tl-none shadow-xs flex items-center gap-1.5">
              <span className="text-xs text-slate-500 font-medium mr-1">Consulting resort database</span>
              <span className="w-1.5 h-1.5 bg-emerald-600 rounded-full animate-bounce"></span>
              <span className="w-1.5 h-1.5 bg-emerald-600 rounded-full animate-bounce [animation-delay:0.2s]"></span>
              <span className="w-1.5 h-1.5 bg-emerald-600 rounded-full animate-bounce [animation-delay:0.4s]"></span>
            </div>
          </div>
        )}

        {/* Error Notification */}
        {errorMessage && (
          <div className="p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between">
            <span>{errorMessage}</span>
            <button
              onClick={() => handleSendMessage()}
              className="flex items-center gap-1 px-2.5 py-1 bg-rose-100 hover:bg-rose-200 text-rose-900 rounded font-semibold text-[11px] transition-colors"
            >
              <RefreshCw className="w-3 h-3" /> Retry
            </button>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Interactive Date Picker Toggle Section */}
      {showDatePicker && (
        <div className="p-3 border-t border-slate-200 bg-emerald-50/50">
          <DateGuestPicker onSearch={handleFormSearch} isLoading={isLoading} />
        </div>
      )}

      {/* Quick Prompt Suggestions */}
      <div className="px-4 py-2 border-t border-slate-100 bg-white">
        <div className="flex items-center gap-1.5 overflow-x-auto no-scrollbar py-1">
          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider shrink-0 flex items-center gap-1">
            <Sparkles className="w-3 h-3 text-amber-500" /> Suggested:
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
              className="text-xs shrink-0 px-2.5 py-1 rounded-full bg-slate-100 hover:bg-emerald-50 hover:text-emerald-800 text-slate-700 border border-slate-200/80 transition-colors cursor-pointer"
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
        className="p-3 border-t border-slate-200 bg-white flex items-center gap-2"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about The Grand Azure Goa or check room availability..."
          disabled={isLoading}
          className="flex-1 px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs sm:text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-emerald-600 focus:bg-white transition-all"
        />

        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="p-2.5 rounded-xl bg-emerald-700 hover:bg-emerald-800 disabled:bg-slate-200 text-white disabled:text-slate-400 transition-colors shadow-xs cursor-pointer"
        >
          <Send className="w-4 h-4" />
        </button>
      </form>
    </div>
  );
}
