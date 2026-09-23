"use client";

import React, { useState } from "react";
import { Calendar, Users, Bed, Check, Sparkles, PhoneCall } from "lucide-react";

export interface RoomItem {
  id: string;
  type: string;
  maxGuests: number;
  bedType: string;
  description: string;
  pricePerNight: number;
  totalPrice: number;
  available: boolean;
}

export interface AvailabilityData {
  available: boolean;
  checkIn?: string;
  checkOut?: string;
  nights?: number;
  adults?: number;
  rooms?: RoomItem[];
  message?: string;
}

interface AvailabilityCardProps {
  data: AvailabilityData;
}

export default function AvailabilityCard({ data }: AvailabilityCardProps) {
  const [reservedRoomId, setReservedRoomId] = useState<string | null>(null);

  if (!data) return null;

  if (!data.available) {
    return (
      <div className="mt-3 p-4 bg-black border border-yellow-500/40 text-yellow-200 text-xs font-mono">
        <div className="flex items-start gap-3">
          <div className="p-2 bg-yellow-400 text-black shrink-0 font-bold">
            <PhoneCall className="w-4 h-4" />
          </div>
          <div>
            <h4 className="font-bold text-yellow-400 uppercase tracking-wider">Inventory Update</h4>
            <p className="mt-1 text-zinc-300 leading-relaxed">
              {data.message || "No inventory available for the requested dates. Please adjust dates or contact front desk."}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-3 space-y-3 font-sans">
      {/* Stay Overview Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-3 bg-black border border-[#222222] text-xs">
        <div className="flex items-center gap-2 text-zinc-200 font-mono">
          <Calendar className="w-3.5 h-3.5 text-yellow-400" />
          <span className="font-semibold text-white">
            {data.checkIn} — {data.checkOut}
          </span>
          <span className="text-yellow-400">•</span>
          <span className="text-zinc-400">
            {data.nights} {data.nights === 1 ? "night" : "nights"}
          </span>
        </div>
        <div className="flex items-center gap-1.5 px-2 py-0.5 bg-[#111111] border border-[#262626] text-yellow-400 text-[11px] font-mono">
          <Users className="w-3 h-3" />
          <span>{data.adults} Guest{data.adults && data.adults > 1 ? "s" : ""}</span>
        </div>
      </div>

      {/* Available Room Tiers Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {data.rooms?.map((room) => {
          const isSelected = reservedRoomId === room.id;
          return (
            <div
              key={room.id}
              className={`p-3.5 border transition-colors flex flex-col justify-between ${
                isSelected
                  ? "bg-[#111111] border-yellow-400"
                  : "bg-black border-[#222222] hover:border-[#333333]"
              }`}
            >
              <div>
                <div className="flex items-start justify-between gap-2">
                  <h4 className="font-bold text-white text-xs sm:text-sm tracking-wide uppercase">
                    {room.type}
                  </h4>
                  <span className="inline-flex items-center px-1.5 py-0.5 text-[10px] font-mono bg-[#141414] text-yellow-400 border border-[#262626] shrink-0">
                    <Users className="w-2.5 h-2.5 mr-1" />
                    Max {room.maxGuests}
                  </span>
                </div>

                {room.bedType && (
                  <div className="flex items-center gap-1.5 text-[11px] text-zinc-400 mt-1.5 font-mono">
                    <Bed className="w-3 h-3 text-zinc-500 shrink-0" />
                    <span className="truncate">{room.bedType}</span>
                  </div>
                )}

                {room.description && (
                  <p className="text-xs text-zinc-300 mt-2 line-clamp-2 leading-relaxed">
                    {room.description}
                  </p>
                )}
              </div>

              <div className="mt-3 pt-3 border-t border-[#1c1c1c]">
                <div className="flex items-baseline justify-between">
                  <div>
                    <span className="text-sm sm:text-base font-bold text-yellow-400 font-mono">
                      ₹{room.pricePerNight?.toLocaleString("en-IN")}
                    </span>
                    <span className="text-[10px] text-zinc-400 font-mono"> / night</span>
                  </div>
                  {data.nights && data.nights > 1 && (
                    <span className="text-[10px] font-mono text-zinc-300 bg-[#141414] px-1.5 py-0.5 border border-[#262626]">
                      Total: ₹{room.totalPrice?.toLocaleString("en-IN")}
                    </span>
                  )}
                </div>

                <button
                  type="button"
                  onClick={() => setReservedRoomId(isSelected ? null : room.id)}
                  className={`mt-2.5 w-full py-2 px-3 text-xs font-mono font-bold uppercase tracking-wider flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                    isSelected
                      ? "bg-yellow-400 text-black hover:bg-yellow-300"
                      : "bg-[#141414] hover:bg-yellow-400 hover:text-black text-white border border-[#262626]"
                  }`}
                >
                  {isSelected ? (
                    <>
                      <Check className="w-3.5 h-3.5" />
                      Selected for Booking
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-3.5 h-3.5 text-yellow-400" />
                      Select Room
                    </>
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


