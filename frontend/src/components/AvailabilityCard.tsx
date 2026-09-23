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
      <div className="mt-3 p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-sm">
        <div className="flex items-start gap-3">
          <div className="p-2 rounded-lg bg-amber-100 text-amber-700 shrink-0">
            <PhoneCall className="w-5 h-5" />
          </div>
          <div>
            <h4 className="font-semibold text-amber-950">Room Notice</h4>
            <p className="mt-1 text-amber-800">{data.message || "No rooms currently available for the selected dates."}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-3 space-y-3">
      {/* Stay Overview Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 p-3 bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200/80 rounded-xl text-xs">
        <div className="flex items-center gap-2 text-emerald-900 font-medium">
          <Calendar className="w-4 h-4 text-emerald-700" />
          <span>
            {data.checkIn} to {data.checkOut}
          </span>
          <span className="text-emerald-500 font-bold">•</span>
          <span>
            {data.nights} {data.nights === 1 ? "night" : "nights"}
          </span>
        </div>
        <div className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-100/70 text-emerald-800 rounded-full font-medium">
          <Users className="w-3.5 h-3.5" />
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
              className={`p-4 rounded-xl border transition-all duration-200 flex flex-col justify-between ${
                isSelected
                  ? "bg-emerald-50/70 border-emerald-500 shadow-md ring-1 ring-emerald-500"
                  : "bg-white border-slate-200 hover:border-slate-300 hover:shadow-sm"
              }`}
            >
              <div>
                <div className="flex items-start justify-between gap-2">
                  <h4 className="font-semibold text-slate-900 text-sm leading-tight">{room.type}</h4>
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium bg-slate-100 text-slate-700 shrink-0">
                    <Users className="w-3 h-3 mr-1" />
                    Max {room.maxGuests}
                  </span>
                </div>

                {room.bedType && (
                  <div className="flex items-center gap-1.5 text-xs text-slate-500 mt-1.5">
                    <Bed className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    <span className="truncate">{room.bedType}</span>
                  </div>
                )}

                {room.description && (
                  <p className="text-xs text-slate-600 mt-2 line-clamp-2 leading-relaxed">
                    {room.description}
                  </p>
                )}
              </div>

              <div className="mt-3 pt-3 border-t border-slate-100">
                <div className="flex items-baseline justify-between">
                  <div>
                    <span className="text-base font-bold text-slate-950">₹{room.pricePerNight?.toLocaleString("en-IN")}</span>
                    <span className="text-[11px] text-slate-500 font-normal"> / night</span>
                  </div>
                  {data.nights && data.nights > 1 && (
                    <span className="text-[11px] font-medium text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded">
                      Total: ₹{room.totalPrice?.toLocaleString("en-IN")}
                    </span>
                  )}
                </div>

                <button
                  type="button"
                  onClick={() => setReservedRoomId(isSelected ? null : room.id)}
                  className={`mt-2.5 w-full py-1.5 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors cursor-pointer ${
                    isSelected
                      ? "bg-emerald-700 text-white hover:bg-emerald-800"
                      : "bg-slate-900 text-white hover:bg-slate-800"
                  }`}
                >
                  {isSelected ? (
                    <>
                      <Check className="w-3.5 h-3.5" />
                      Selected for Booking
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-3.5 h-3.5 text-amber-300" />
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
