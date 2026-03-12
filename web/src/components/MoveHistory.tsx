"use client";

import { useEffect, useRef } from "react";

interface MoveHistoryProps {
  moves: string[];
}

export default function MoveHistory({ moves }: MoveHistoryProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [moves.length]);

  const pairs: [number, string, string | undefined][] = [];
  for (let i = 0; i < moves.length; i += 2) {
    pairs.push([Math.floor(i / 2) + 1, moves[i], moves[i + 1]]);
  }

  return (
    <div className="bg-zinc-800 rounded p-3 h-64 overflow-y-auto text-sm font-mono">
      {pairs.length === 0 && (
        <p className="text-zinc-500 text-center mt-4">No moves yet</p>
      )}
      {pairs.map(([num, white, black]) => (
        <div key={num} className="flex gap-2 py-0.5">
          <span className="text-zinc-500 w-8 text-right">{num}.</span>
          <span className="w-16">{white}</span>
          <span className="w-16 text-zinc-300">{black || ""}</span>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
