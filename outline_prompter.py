# outline_to_ollama.py
# Simple loop: outline beat -> baseline RAG context -> prompt -> Ollama(qwen3:14b)
# Requires: baseline_rag.py (get_baseline_context), FAISS index already built.

import re, time
from pathlib import Path
import ollama

# ---------- CONFIG ----------
MODEL = "qwen3:8b"
OUTLINE_FILE = Path("outline.txt")     # your outline input
OUT_DIR = Path("outline_outputs")      # where to save model outputs
TOP_K_BASELINE = 5                     # RAG: how many baseline snippets to pull
SLEEP_BETWEEN_CALLS = 0.05             # be nice to your CPU/GPU
TEMPERATURE = 0.2                      # small creativity for prose; drop to 0.0 for rigid tasks

# What you want the model to do with each beat. Change as needed.
TASK_INSTRUCTION = (
    "Expand the outline beat into a short scene sketch (5–8 sentences). "
    "Stay faithful to the beat and the baseline context. Do not invent canon that contradicts context. "
    "Prefer understated prose and implication over explicit tells."
)

# Optional system prompt for tone/constraints (kept neutral)
SYSTEM_PROMPT = (
    "You are a precise scene planner. You use context to maintain continuity. "
    "You write concise, concrete, non-redundant notes."
)

# ---------- HELPERS ----------
def _remove_think_blocks(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()

def _split_outline(text: str) -> list[str]:
    """
    Split outline into beats.
    Rules:
      - split on blank lines
      - also split on headings that start with '#', '##', numbered lines like '1).' or '1)'
      - keep non-empty trimmed chunks
    """
    # First split on double newlines
    chunks = re.split(r"\n\s*\n", text.strip())
    beats = []
    for ch in chunks:
        # Further split if there are clear headings within a chunk
        parts = re.split(r"(?m)^(?:#{1,6}\s+|\d+\)\s+|\d+\.\s+|-+\s*$)", ch.strip())
        for p in parts:
            p = p.strip()
            if p:
                beats.append(p)
    return beats

# ---------- RAG ----------
# Uses the helper from earlier. Make sure baseline_rag.py is in the same folder.
from baseline_rag import get_baseline_context

def build_prompt(beat_text: str, baseline_ctx: str) -> list[dict]:
    """Return messages list for ollama.chat."""
    ctx_block = ""
    if baseline_ctx.strip():
        ctx_block = "\n\nBaseline context (trusted facts; if conflict, prefer the beat unless it's canon):\n" + baseline_ctx
    user = (
        f"{TASK_INSTRUCTION}\n\n"
        f"Outline beat:\n{beat_text}\n"
        f"{ctx_block}\n\n"
        "Output: a focused scene sketch. Avoid florid language; emphasize concrete actions, decisions, and shifts."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user}
    ]

def call_model(messages: list[dict]) -> str:
    res = ollama.chat(model=MODEL, messages=messages, options={"temperature": TEMPERATURE})
    content = _remove_think_blocks(res["message"]["content"]).strip()
    return content

# ---------- MAIN ----------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = OUTLINE_FILE.read_text(encoding="utf-8-sig")
    beats = _split_outline(text)
    if not beats:
        print(f"No beats found in {OUTLINE_FILE.resolve()}")
        return

    for i, beat in enumerate(beats, start=1):
        # Retrieve baseline snippets to ground the beat
        baseline_ctx = get_baseline_context(beat, top_k=TOP_K_BASELINE)

        # Build messages and query the model
        msgs = build_prompt(beat, baseline_ctx)
        print(msgs)
        try:
            reply = call_model(msgs)
        except Exception as e:
            print(f"[ERROR] beat {i}: {e}")
            continue

        # Save output
        out_path = OUT_DIR / f"beat-{i:03d}.md"
        header = f"# Beat {i:03d}\n\n## Outline\n{beat}\n\n## Output\n"
        out_path.write_text(header + reply + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")

        time.sleep(SLEEP_BETWEEN_CALLS)

if __name__ == "__main__":
    main()
