export const SYSTEM_PROMPT =
  "You are a chess engine. Given the current board position in FEN notation " +
  "and the list of legal moves, select the best move in UCI notation.\n\n" +
  "Respond with your chosen move inside <move> tags. " +
  "For example: <move>e2e4</move>\n\n" +
  "You may optionally reason about the position before providing your move.";

const MOVE_REGEX = /<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*<\/move>/;

export function extractMove(text: string): string | null {
  const match = text.match(MOVE_REGEX);
  return match ? match[1] : null;
}
