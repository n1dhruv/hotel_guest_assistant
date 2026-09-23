"use client";

import React, { useState } from "react";
import { Calendar, Users, Search } from "lucide-react";

interface DateGuestPickerProps {
  onSearch: (checkIn: string, checkOut: string, adults: number) => void;
  isLoading?: boolean;
}

export default function DateGuestPicker({ onSearch, isLoading = false }: DateGuestPickerProps) {
  // Helper to format Date to YYYY-MM-DD
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
      className="p-3.5 bg-slate-50 border border-slate-200/90 rounded-xl space-y-3 text-xs"
    >
      <div className="flex items-center gap-1.5 font-semibold text-slate-800">
        <Calendar className="w-4 h-4 text-emerald-700" />
        <span>Select Dates & Guest Count</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
        {/* Check-in Date */}
        <div>
          <label className="block text-[11px] font-medium text-slate-600 mb-1">
            Check-in Date
          </label>
          <input
            type="date"
            min={formatISO(today)}
            value={checkIn}
            onChange={handleCheckInChange}
            required
            className="w-full px-2.5 py-1.5 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-600 text-xs"
          />
        </div>

        {/* Check-out Date */}
        <div>
          <label className="block text-[11px] font-medium text-slate-600 mb-1">
            Check-out Date
          </label>
          <input
            type="date"
            min={checkIn || formatISO(today)}
            value={checkOut}
            onChange={(e) => setCheckOut(e.target.value)}
            required
            className="w-full px-2.5 py-1.5 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-600 text-xs"
          />
        </div>

        {/* Guest Count */}
        <div>
          <label className="block text-[11px] font-medium text-slate-600 mb-1">
            Adult Guests
          </label>
          <select
            value={adults}
            onChange={(e) => setAdults(Number(e.target.value))}
            className="w-full px-2.5 py-1.5 bg-white border border-slate-300 rounded-lg text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-600 text-xs"
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
        className="w-full py-2 px-3 bg-emerald-700 hover:bg-emerald-800 disabled:bg-slate-400 text-white font-medium rounded-lg flex items-center justify-center gap-1.5 transition-colors cursor-pointer text-xs"
      >
        <Search className="w-3.5 h-3.5" />
        {isLoading ? "Checking Live Inventory..." : "Check Availability & Rates"}
      </button>
    </form>
  );
}
