import OpenAI from "openai";
import { SYSTEM_PROMPT, extractMove } from "./constants";

const client = new OpenAI({
  baseURL: process.env.VLLM_BASE_URL || "http://localhost:8000/v1",
  apiKey: "not-needed",
});

export async function getModelMove(
  fen: string,
  legalMoves: string[]
): Promise<{ move: string | null; reasoning: string }> {
  const messages: OpenAI.ChatCompletionMessageParam[] = [
    { role: "system", content: SYSTEM_PROMPT },
    {
      role: "user",
      content: `Position (FEN): ${fen}\nLegal moves: ${legalMoves.join(" ")}`,
    },
  ];

  for (const temp of [0.3, 0.6]) {
    try {
      const response = await client.chat.completions.create({
        model: process.env.VLLM_MODEL_NAME || "default",
        messages,
        max_tokens: 256,
        temperature: temp,
      });

      const raw = response.choices[0].message.content || "";
      const move = extractMove(raw);

      if (move) {
        const reasoning = raw.replace(/<move>.*<\/move>/, "").trim();
        return { move, reasoning };
      }
    } catch (e) {
      if (temp === 0.6) throw e;
    }
  }

  return { move: null, reasoning: "" };
}
