"use client";

import { useState, useCallback, useRef } from "react";
import { Chess } from "chess.js";
import { toast } from "sonner";
import Board from "./Board";
import MoveHistory from "./MoveHistory";
import GameStatus from "./GameStatus";

type GameResult = "playing" | "checkmate-win" | "checkmate-loss" | "draw";

export default function ChessGame() {
  const gameRef = useRef(new Chess());
  const [fen, setFen] = useState(gameRef.current.fen());
  const [isThinking, setIsThinking] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const [result, setResult] = useState<GameResult>("playing");

  const checkGameOver = useCallback((game: Chess) => {
    if (game.isCheckmate()) {
      setResult(game.turn() === "b" ? "checkmate-win" : "checkmate-loss");
    } else if (game.isDraw() || game.isStalemate() || game.isThreefoldRepetition() || game.isInsufficientMaterial()) {
      setResult("draw");
    }
  }, []);

  const requestModelMove = useCallback(async (game: Chess) => {
    setIsThinking(true);

    try {
      const res = await fetch("/api/move", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fen: game.fen() }),
      });

      const data = await res.json();

      if (!res.ok) {
        if (res.status === 503) {
          toast.error("Model server unreachable", {
            description: "Ensure vLLM is running and rv forward is active.",
          });
        } else if (res.status === 422) {
          toast.error("Model returned an invalid move", {
            description: data.error,
          });
        } else {
          toast.error(data.error || "Something went wrong");
        }
        return;
      }

      game.move({
        from: data.move.slice(0, 2),
        to: data.move.slice(2, 4),
        promotion: data.move[4] || undefined,
      });

      setFen(game.fen());
      setHistory(game.history());
      checkGameOver(game);
    } catch {
      toast.error("Connection failed", {
        description: "Could not reach the API. Is the dev server running?",
      });
    } finally {
      setIsThinking(false);
    }
  }, [checkGameOver]);

  const onPieceDrop = useCallback(
    (sourceSquare: string, targetSquare: string): boolean => {
      if (isThinking || result !== "playing") return false;

      const game = gameRef.current;
      try {
        game.move({
          from: sourceSquare,
          to: targetSquare,
          promotion: "q",
        });
      } catch {
        return false;
      }

      setFen(game.fen());
      setHistory(game.history());
      checkGameOver(game);

      if (!game.isGameOver()) {
        requestModelMove(game);
      }

      return true;
    },
    [isThinking, result, checkGameOver, requestModelMove]
  );

  const newGame = useCallback(() => {
    const game = gameRef.current;
    game.reset();
    setFen(game.fen());
    setHistory([]);
    setResult("playing");
    setIsThinking(false);
  }, []);

  return (
    <div className="flex flex-col lg:flex-row gap-6 items-start">
      <Board
        fen={fen}
        onPieceDrop={onPieceDrop}
        disabled={isThinking || result !== "playing"}
      />
      <div className="flex flex-col gap-4 w-full lg:w-64">
        <GameStatus
          result={result}
          isThinking={isThinking}
          isCheck={gameRef.current.isCheck()}
          turn={gameRef.current.turn()}
        />
        <MoveHistory moves={history} />
        <button
          onClick={newGame}
          className="px-4 py-2 bg-zinc-700 hover:bg-zinc-600 rounded text-sm font-medium transition-colors cursor-pointer"
        >
          New Game
        </button>
      </div>
    </div>
  );
}
