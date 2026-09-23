"use client";

import React, { useState } from "react";
import { Calendar, Search } from "lucide-react";

interface DateGuestPickerProps {
  onSearch: (checkIn: string, checkOut: string, adults: number) => void;
  isLoading?: boolean;
}

export default function DateGuestPicker({ onSearch, isLoading = false }: DateGuestPickerProps) {
  const formatISO = (d: Date) => d.toISOString().split("T")[0];

  const today = new Date();
  const defaultCheckIn = new Date();
  defaultCheckIn.setDate(today.getDate() + 5);

  const defaultCheckOut = new Date();
  defaultCheckOut.setDate(defaultCheckIn.getDate() + 2);

  const [checkIn, setCheckIn] = useState<string>(formatISO(defaultCheckIn));
  const [checkOut, setCheckOut] = useState<string>(formatISO(defaultCheckOut));
  const [adults, setAdults] = useState<number>(2);

  const handleCheckInChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setCheckIn(val);
    if (val && checkOut && val >= checkOut) {
      const nextDay = new Date(val);
      nextDay.setDate(nextDay.getDate() + 1);
      setCheckOut(formatISO(nextDay));
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!checkIn || !checkOut) return;
    onSearch(checkIn, checkOut, adults);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="p-3.5 bg-black border border-[#222222] space-y-3 text-xs"
    >
      <div className="flex items-center justify-between border-b border-[#1c1c1c] pb-2">
        <div className="flex items-center gap-2 font-mono uppercase tracking-wider text-yellow-400 font-bold text-xs">
          <Calendar className="w-3.5 h-3.5" />
          <span>Select Dates & Guests</span>
        </div>
        <span className="text-[10px] font-mono text-zinc-400 uppercase">Live Inventory</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
        <div>
          <label className="block text-[10px] font-mono uppercase tracking-wider text-zinc-400 mb-1">
            Check-In
          </label>
          <input
            type="date"
            min={formatISO(today)}
            value={checkIn}
            onChange={handleCheckInChange}
            required
            className="w-full px-2.5 py-1.5 bg-black border border-[#262626] text-white focus:border-yellow-400 outline-none text-xs font-mono"
          />
        </div>

        <div>
          <label className="block text-[10px] font-mono uppercase tracking-wider text-zinc-400 mb-1">
            Check-Out
          </label>
          <input
            type="date"
            min={checkIn || formatISO(today)}
            value={checkOut}
            onChange={(e) => setCheckOut(e.target.value)}
            required
            className="w-full px-2.5 py-1.5 bg-black border border-[#262626] text-white focus:border-yellow-400 outline-none text-xs font-mono"
          />
        </div>

        <div>
          <label className="block text-[10px] font-mono uppercase tracking-wider text-zinc-400 mb-1">
            Adult Guests
          </label>
          <select
            value={adults}
            onChange={(e) => setAdults(Number(e.target.value))}
            className="w-full px-2.5 py-1.5 bg-black border border-[#262626] text-white focus:border-yellow-400 outline-none text-xs font-mono"
          >
            <option value={1}>1 Adult</option>
            <option value={2}>2 Adults</option>
            <option value={3}>3 Adults (Deluxe / Suite)</option>
            <option value={4}>4 Adults (Family Suite)</option>
            <option value={5}>5 Adults (Maharaja Suite)</option>
          </select>
        </div>
      </div>


      <button
        type="submit"
        disabled={isLoading}
        className="w-full py-2 px-3 bg-yellow-400 hover:bg-yellow-300 disabled:bg-[#202026] text-black disabled:text-zinc-600 font-bold uppercase tracking-wider flex items-center justify-center gap-1.5 transition-colors cursor-pointer text-xs font-mono"
      >
        <Search className="w-3.5 h-3.5" />
        {isLoading ? "Querying Room Database..." : "Check Availability & Tariffs"}
      </button>
    </form>
  );
}

