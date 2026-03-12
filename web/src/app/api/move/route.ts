import { Chess } from "chess.js";
import { getModelMove } from "@/lib/chess-ai";

export async function POST(request: Request) {
  let body: { fen?: string };
  try {
    body = await request.json();
  } catch {
    return Response.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { fen } = body;
  if (!fen) {
    return Response.json({ error: "Missing fen" }, { status: 400 });
  }

  const game = new Chess();
  try {
    game.load(fen);
  } catch {
    return Response.json({ error: "Invalid FEN" }, { status: 400 });
  }

  if (game.isGameOver()) {
    return Response.json({ error: "Game is already over" }, { status: 400 });
  }

  const legalMoves = game.moves({ verbose: true }).map((m) => {
    return m.from + m.to + (m.promotion || "");
  });

  let result;
  try {
    result = await getModelMove(fen, legalMoves);
  } catch {
    return Response.json(
      { error: "Cannot reach model server. Ensure rv forward 8000 chess-serve is running." },
      { status: 503 }
    );
  }

  if (!result.move) {
    return Response.json(
      { error: "Model failed to produce a valid move" },
      { status: 422 }
    );
  }

  try {
    const moveResult = game.move({
      from: result.move.slice(0, 2),
      to: result.move.slice(2, 4),
      promotion: result.move[4] || undefined,
    });
    if (moveResult) {
      return Response.json({
        move: result.move,
        san: moveResult.san,
        reasoning: result.reasoning,
      });
    }
  } catch {
    // fall through
  }

  return Response.json(
    { error: "Model returned an illegal move" },
    { status: 422 }
  );
}
