"use client";

interface GameStatusProps {
  result: "playing" | "checkmate-win" | "checkmate-loss" | "draw";
  isThinking: boolean;
  isCheck: boolean;
  turn: "w" | "b";
}

export default function GameStatus({ result, isThinking, isCheck, turn }: GameStatusProps) {
  let message: string;
  let style = "text-zinc-300";

  switch (result) {
    case "checkmate-win":
      message = "Checkmate — You win!";
      style = "text-green-400 font-bold";
      break;
    case "checkmate-loss":
      message = "Checkmate — Model wins!";
      style = "text-red-400 font-bold";
      break;
    case "draw":
      message = "Draw";
      style = "text-yellow-400 font-bold";
      break;
    default:
      if (isThinking) {
        message = "Model is thinking\u2026";
      } else if (isCheck) {
        message = "Check! Your turn";
      } else {
        message = turn === "w" ? "Your turn (White)" : "Your turn (Black)";
      }
  }

  return (
    <div className="bg-zinc-800 rounded p-3">
      <p className={style}>{message}</p>
      {isThinking && (
        <div className="flex gap-1 mt-2">
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className="w-2 h-2 bg-zinc-400 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 150}ms` }}
            />
          ))}
        </div>
      )}
    </div>
  );
}
