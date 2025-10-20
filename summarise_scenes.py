import json, re, time, uuid
from pathlib import Path
from typing import Dict, Any
import ollama

# ---------------- CONFIG ----------------
MODEL = "qwen3:8b"
CHUNKS_DIR = Path("text_chunks")
OUT_DIR = Path("memories")

# ---------------- HELPERS ----------------
def _remove_think_blocks(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s

def ask_ollama_for_summary(model: str, scene_text: str,
                           retries: int = 2, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Ask model for a short summary memory of the scene.
    """
    system = (
        "Summarise a fiction scene into one short memory entry for later reference. "
        "Return ONLY JSON with fields: id, summary, rationale, confidence (0-1). "
        "- 'summary': 1–3 sentences, <= 70 words, describing the key event(s). "
        "- 'rationale': why this summary captures the scene. "
        "- 'confidence': float 0.0–1.0."
    )
    prompt = (
        "Scene:\n" + scene_text[:8000] +
        "\n\nRules:\n"
        "- Focus on what happened, not minor details.\n"
        "- Be concise but concrete.\n"
        "- If the scene is mostly descriptive or setup, capture that tone."
    )

    for attempt in range(retries + 1):
        res = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        content = _remove_think_blocks(res["message"]["content"])
        content = _strip_code_fences(content)

        try:
            obj = json.loads(content)
            return {
                "id": str(uuid.uuid4()),
                "summary": obj.get("summary"),
                "rationale": obj.get("rationale"),
                "confidence": float(obj.get("confidence", 0))
            }
        except Exception:
            if attempt == retries:
                raise ValueError(f"Model did not return valid JSON:\n{content}")
            time.sleep(0.5)

    return {"id": str(uuid.uuid4()), "summary": None,
            "rationale": "parse error", "confidence": 0.0}

# ---------------- MAIN ----------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(CHUNKS_DIR.glob("scene-*.txt"))
    if not files:
        print(f"No chunk files found in {CHUNKS_DIR.resolve()}")
        return

    for f in files:
        scene_text = f.read_text(encoding="utf-8-sig").strip()
        if not scene_text:
            continue

        try:
            summary = ask_ollama_for_summary(MODEL, scene_text)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
            continue

        payload: Dict[str, Any] = {
            "file": str(f),
            "model": MODEL,
            "scene": f.stem,
            **summary
        }
        out_dir =  OUT_DIR / 'summaries'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / 'summaries' /(f.stem + "-summary.json")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

        time.sleep(0.1)

if __name__ == "__main__":
    main()
