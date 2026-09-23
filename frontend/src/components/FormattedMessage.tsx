import React from "react";

interface FormattedMessageProps {
  content: string;
  isUser?: boolean;
}

export default function FormattedMessage({ content, isUser = false }: FormattedMessageProps) {
  if (!content) return null;

  const rawLines = content.split("\n");

  type Block =
    | { type: "paragraph"; lines: string[] }
    | { type: "bullet-list"; items: string[] }
    | { type: "numbered-list"; items: string[] }
    | { type: "heading"; level: number; text: string }
    | { type: "divider" };

  const blocks: Block[] = [];
  let currentParagraph: string[] = [];
  let currentList: { type: "bullet-list" | "numbered-list"; items: string[] } | null = null;

  const flushParagraph = () => {
    if (currentParagraph.length > 0) {
      blocks.push({ type: "paragraph", lines: [...currentParagraph] });
      currentParagraph = [];
    }
  };

  const flushList = () => {
    if (currentList) {
      blocks.push({ type: currentList.type, items: [...currentList.items] });
      currentList = null;
    }
  };

  for (let i = 0; i < rawLines.length; i++) {
    const line = rawLines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    if (trimmed === "---" || trimmed === "***") {
      flushParagraph();
      flushList();
      blocks.push({ type: "divider" });
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      blocks.push({
        type: "heading",
        level: headingMatch[1].length,
        text: headingMatch[2]
      });
      continue;
    }

    const bulletMatch = trimmed.match(/^[*•-]\s+(.*)$/);
    if (bulletMatch) {
      flushParagraph();
      if (!currentList || currentList.type !== "bullet-list") {
        flushList();
        currentList = { type: "bullet-list", items: [] };
      }
      currentList.items.push(bulletMatch[1]);
      continue;
    }

    const numberedMatch = trimmed.match(/^\d+\.\s+(.*)$/);
    if (numberedMatch) {
      flushParagraph();
      if (!currentList || currentList.type !== "numbered-list") {
        flushList();
        currentList = { type: "numbered-list", items: [] };
      }
      currentList.items.push(numberedMatch[1]);
      continue;
    }

    flushList();
    currentParagraph.push(line);
  }

  flushParagraph();
  flushList();

  return (
    <div className={`space-y-2.5 ${isUser ? "text-white" : "text-zinc-200"}`}>
      {blocks.map((block, idx) => {
        if (block.type === "divider") {
          return <hr key={idx} className="border-[#222222] my-3" />;
        }

        if (block.type === "heading") {
          if (block.level === 1) {
            return (
              <h2 key={idx} className="font-bold text-white text-base tracking-wide mt-2">
                {renderInline(block.text, isUser)}
              </h2>
            );
          }
          if (block.level === 2) {
            return (
              <h3 key={idx} className="font-bold text-yellow-400 text-sm tracking-wider uppercase mt-2">
                {renderInline(block.text, isUser)}
              </h3>
            );
          }
          return (
            <h4 key={idx} className="font-bold text-white text-xs tracking-wider uppercase mt-1">
              {renderInline(block.text, isUser)}
            </h4>
          );
        }

        if (block.type === "bullet-list") {
          return (
            <ul key={idx} className="space-y-2 my-2.5 pl-0.5">
              {block.items.map((item, itemIdx) => (
                <li key={itemIdx} className="flex items-start gap-2.5">
                  <span
                    className={`w-1.5 h-1.5 mt-1.5 shrink-0 ${
                      isUser ? "bg-white" : "bg-yellow-400"
                    }`}
                  />
                  <div className="flex-1 leading-relaxed">
                    {renderInline(item, isUser)}
                  </div>
                </li>
              ))}
            </ul>
          );
        }

        if (block.type === "numbered-list") {
          return (
            <ol key={idx} className="space-y-2 my-2.5 pl-0.5">
              {block.items.map((item, itemIdx) => (
                <li key={itemIdx} className="flex items-start gap-2">
                  <span
                    className={`font-mono text-xs font-bold shrink-0 ${
                      isUser ? "text-white" : "text-yellow-400"
                    }`}
                  >
                    {itemIdx + 1}.
                  </span>
                  <div className="flex-1 leading-relaxed">
                    {renderInline(item, isUser)}
                  </div>
                </li>
              ))}
            </ol>
          );
        }

        return (
          <p key={idx} className="leading-relaxed">
            {block.lines.map((line, lineIdx) => (
              <React.Fragment key={lineIdx}>
                {renderInline(line, isUser)}
                {lineIdx < block.lines.length - 1 && <br />}
              </React.Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}

// Helper to parse inline bold (**text**), italics (*text*), and code (`text`)
function renderInline(text: string, isUser: boolean) {
  const parts = text.split(/(\*\*\*.*?\*\*\*|\*\*.*?\*\*|\*[^*\n]+?\*|`.*?`)/g);

  return parts.map((part, index) => {
    // Bold + Italic: ***text***
    if (part.startsWith("***") && part.endsWith("***") && part.length >= 6) {
      const inner = part.slice(3, -3);
      return (
        <strong
          key={index}
          className={`font-bold italic tracking-wide ${
            isUser ? "text-yellow-300" : "text-white"
          }`}
        >
          {inner}
        </strong>
      );
    }

    // Bold: **text**
    if (part.startsWith("**") && part.endsWith("**") && part.length >= 4) {
      const inner = part.slice(2, -2);
      return (
        <strong
          key={index}
          className={`font-bold tracking-wide ${
            isUser ? "text-yellow-300" : "text-white font-bold"
          }`}
        >
          {inner}
        </strong>
      );
    }

    // Italic: *text*
    if (part.startsWith("*") && part.endsWith("*") && part.length >= 2) {
      const inner = part.slice(1, -1);
      return (
        <em key={index} className="italic text-zinc-300">
          {inner}
        </em>
      );
    }

    // Inline code: `text`
    if (part.startsWith("`") && part.endsWith("`") && part.length >= 2) {
      const inner = part.slice(1, -1);
      return (
        <code
          key={index}
          className="font-mono text-[11px] px-1 py-0.5 bg-[#18181f] border border-[#2a2a34] text-yellow-400 mx-0.5"
        >
          {inner}
        </code>
      );
    }

    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}
