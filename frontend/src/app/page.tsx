import React from "react";
import ChatInterface from "@/components/ChatInterface";

export default function Home() {
  return (
    <main className="h-screen w-screen overflow-hidden bg-black text-[#f4f4f5] flex items-center justify-center p-2 sm:p-4 md:p-6">
      <div className="w-full max-w-4xl h-[95vh] flex flex-col">
        <ChatInterface />
      </div>
    </main>
  );
}


