"use client";

import { Chessboard } from "react-chessboard";

interface BoardProps {
  fen: string;
  onPieceDrop: (source: string, target: string) => boolean;
  disabled: boolean;
}

export default function Board({ fen, onPieceDrop, disabled }: BoardProps) {
  return (
    <div className="shrink-0" style={{ width: 480, height: 480 }}>
      <Chessboard
        options={{
          position: fen,
          onPieceDrop: ({ sourceSquare, targetSquare }) =>
            onPieceDrop(sourceSquare, targetSquare ?? ""),
          animationDurationInMs: 200,
          allowDragging: !disabled,
          boardStyle: {
            borderRadius: "4px",
            boxShadow: "0 4px 12px rgba(0,0,0,0.4)",
          },
          darkSquareStyle: { backgroundColor: "#779952" },
          lightSquareStyle: { backgroundColor: "#edeed1" },
        }}
      />
    </div>
  );
}
