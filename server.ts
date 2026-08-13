import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // Initialize Gemini AI Client
  const getAi = () => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is not configured in environment.");
    }
    return new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  };

  // API endpoint: Generate AI Birthday Wish
  app.post("/api/generate-wish", async (req, res) => {
    try {
      const { recipientName, age, relationship, tone, hobbies } = req.body;

      if (!recipientName) {
        return res.status(400).json({ error: "Recipient name is required" });
      }

      const ai = getAi();
      const prompt = `Write a creative, heartwarming, and highly memorable birthday message for ${recipientName}.
Details:
- Age / Milestone: ${age || "a magical milestone year"}
- Relationship: ${relationship || "a dear friend"}
- Tone: ${tone || "Heartfelt & Inspiring"} (Options like Heartfelt, Hilarious Roast, Epic Superhero, Rhyming Poem, Nostalgic, Magical)
- Favorite Hobbies/Traits: ${hobbies || "bringing joy to everyone"}

Guidelines:
1. Include a short catchy title (e.g., "To the Legend Sarah! ✨").
2. Write 2-3 engaging paragraphs or stanzas of message.
3. Include 3 custom "Birthday Superpowers" or fun compliments tailored to them.
4. Keep it uplifting, warm, and fun!

Respond in JSON with this exact structure:
{
  "title": "...",
  "message": "...",
  "compliments": ["...", "...", "..."],
  "signature": "With endless love & sparkles 💖"
}`;

      const response = await ai.models.generateContent({
        model: "gemini-3.6-flash",
        contents: prompt,
        config: {
          responseMimeType: "application/json",
        },
      });

      const responseText = response.text || "{}";
      const data = JSON.parse(responseText);

      return res.json({ success: true, wish: data });
    } catch (error: any) {
      console.error("Error generating wish:", error);
      return res.status(500).json({
        success: false,
        error: error.message || "Failed to generate birthday wish",
      });
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Birthday Wish server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
