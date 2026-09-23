import React from "react";
import ChatInterface from "@/components/ChatInterface";

export default function Home() {
  return (
    <main className="h-[100dvh] w-full overflow-hidden bg-black text-[#f4f4f5] flex items-center justify-center p-0 sm:p-3 md:p-6">
      {/* Full screen on mobile, elegant container on tablet/desktop */}
      <div className="w-full max-w-4xl h-full sm:h-[94vh] flex flex-col">
        <ChatInterface />
      </div>
    </main>
  );
}





