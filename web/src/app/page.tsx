import ChessGame from "@/components/ChessGame";

export default function Home() {
  return (
    <main className="min-h-screen bg-zinc-900 text-white flex flex-col items-center justify-center p-6">
      <h1 className="text-2xl font-bold mb-1">Play Chess vs RL-Trained Qwen 7B</h1>
      <p className="text-zinc-400 mb-8 text-sm">
        GRPO-trained model &mdash; 100% legal moves, 90% win rate vs GPT-4o
      </p>
      <ChessGame />
    </main>
  );
}
