import React from "react";
import { Star, MapPin, Phone, Mail, Clock, Waves, Coffee, ShieldCheck, Zap } from "lucide-react";
import ChatInterface from "@/components/ChatInterface";

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-900 to-slate-950 text-slate-100 flex flex-col">
      {/* Top Resort Banner Header */}
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-amber-700 flex items-center justify-center font-serif font-bold text-white shadow-md text-lg">
              GA
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <h1 className="font-serif text-base sm:text-lg font-semibold tracking-wide text-white">
                  The Grand Azure
                </h1>
                <span className="text-amber-400 flex items-center text-xs ml-1">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} className="w-3 h-3 fill-amber-400" />
                  ))}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 flex items-center gap-1">
                <MapPin className="w-3 h-3 text-amber-500" /> Candolim Beach, North Goa, India
              </p>
            </div>
          </div>

          {/* Quick Contact & Status Badges */}
          <div className="flex items-center gap-4 text-xs">
            <a
              href="tel:+918322498000"
              className="hidden md:flex items-center gap-1.5 text-slate-300 hover:text-white transition-colors"
            >
              <Phone className="w-3.5 h-3.5 text-emerald-400" />
              <span>+91 832 249 8000</span>
            </a>
            <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/80 border border-emerald-500/30 text-emerald-300 text-[11px] font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span>AI Concierge Live</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <div className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column: Property Summary & Key Amenities (Hidden on mobile or compact) */}
        <aside className="lg:col-span-4 space-y-4">
          <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700/60 backdrop-blur-xs">
            <span className="text-[11px] uppercase tracking-widest text-amber-400 font-semibold">
              Luxury Beachfront Hospitality
            </span>
            <h2 className="font-serif text-xl font-bold text-white mt-1">
              Your Personal Stay Assistant
            </h2>
            <p className="text-xs text-slate-300 mt-2 leading-relaxed">
              Ask any question about our Candolim Beach resort, policies, and amenities, or check real-time room availability and tariffs.
            </p>

            <div className="mt-4 pt-4 border-t border-slate-700/60 space-y-3 text-xs">
              <div className="flex items-start gap-2.5">
                <Clock className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <div className="font-medium text-white">Check-in / Check-out</div>
                  <div className="text-slate-400 text-[11px]">2:00 PM IST Check-in • 11:00 AM IST Check-out</div>
                </div>
              </div>

              <div className="flex items-start gap-2.5">
                <Waves className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                <div>
                  <div className="font-medium text-white">Oceanview Infinity Pool</div>
                  <div className="text-slate-400 text-[11px]">Heated pool open daily 6:00 AM – 9:00 PM</div>
                </div>
              </div>

              <div className="flex items-start gap-2.5">
                <Coffee className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                <div>
                  <div className="font-medium text-white">Complimentary Breakfast</div>
                  <div className="text-slate-400 text-[11px]">Royal buffet with Pure Veg & Jain counters</div>
                </div>
              </div>

              <div className="flex items-start gap-2.5">
                <Zap className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
                <div>
                  <div className="font-medium text-white">EV Charging & Valet</div>
                  <div className="text-slate-400 text-[11px]">Tata Power & universal fast chargers on site</div>
                </div>
              </div>

              <div className="flex items-start gap-2.5">
                <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <div className="font-medium text-white">Mandatory Check-in ID</div>
                  <div className="text-slate-400 text-[11px]">Aadhaar / Passport / Voter ID required by law</div>
                </div>
              </div>
            </div>
          </div>

          {/* Concierge Desk Card */}
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs space-y-2">
            <div className="font-semibold text-slate-200">Front Desk & Concierge</div>
            <div className="text-slate-400 text-[11px] space-y-1">
              <div className="flex items-center gap-1.5">
                <Phone className="w-3.5 h-3.5 text-slate-500" />
                <span>+91 832 249 8000 (24/7)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Mail className="w-3.5 h-3.5 text-slate-500" />
                <span>concierge@grandazuregoa.com</span>
              </div>
            </div>
          </div>
        </aside>

        {/* Right Column: AI Assistant Chat Interface */}
        <section className="lg:col-span-8">
          <ChatInterface />
        </section>
      </div>

      {/* Footer */}
      <footer className="mt-auto border-t border-slate-800/80 bg-slate-950 py-4 text-center text-xs text-slate-500">
        <p>© 2026 The Grand Azure Heritage Resort & Spa. Candolim, Goa, India. Powered by Full-Stack AI Assistant.</p>
      </footer>
    </main>
  );
}
