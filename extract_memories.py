import json, re, time, uuid
from pathlib import Path
from typing import Dict, Any
import ollama

# ---------------- CONFIG ----------------
MODEL = "qwen3:8b"
LABELS_DIR = Path("labels")
CHUNKS_DIR = Path("text_chunks")
OUT_DIR = Path("memories")

TAGS_WITH_DESC = {
    "important_memory_for_character": "foundational memories that might shape the character in the future",
    "relationship_change": "important changes to status quo between characters",
    "plot_fact": "an important happenstance that might be later referenced by someone",
    "magic_system": "concrete rules and explanations for how fantastical elements operate within the story",
    "fauna and flora": "names, species, physical descriptions and anything that is necessary to keep consistency across the story when referencing fauna and flora",
    "location_fact": "structural details about scenery that are necessary to keep consistency across the story when referencing past locations",
    "timeline": "mentions of time passing, in regards to the current in-world date, within the constraints of the scene",
    "foreshadowing": "long-term foreshadowing elements that should later be touched upon and utilised"
}
# ----------------------------------------

def _remove_think_blocks(s: str) -> str:
    return re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL | re.IGNORECASE).strip()

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    return s

def ask_ollama_for_memory(model: str, scene_text: str, tag: str,
                          retries: int = 2, temperature: float = 0.1) -> Dict[str, Any]:
    """Ask model to extract a concrete memory for a given tag from the scene text."""
    system = (
        "You extract a SINGLE concrete memory for the given tag from the scene text. "
        "Return ONLY JSON with fields: memory, rationale, confidence (0-1). "
        "If no valid info exists for this tag, return memory=null, rationale='no data', confidence=0.0."
    )
    prompt = (
        f"Tag: {tag}\n\n"
        f"this tag refers to {TAGS_WITH_DESC[f'{tag}']}"
        f"Scene:\n{scene_text[:8000]}\n\n"
        "Rules:\n"
        "- Memory must be 1–2 sentences, <= 45 words.\n"
        "- Be specific and verifiable (names, events, places, numbers if present).\n"
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
                "tag": tag,
                "memory": obj.get("memory"),
                "rationale": obj.get("rationale"),
                "confidence": float(obj.get("confidence", 0)),
            }
        except Exception:
            if attempt == retries:
                raise ValueError(f"Model did not return valid JSON:\n{content}")
            time.sleep(0.5)
    return {"id": str(uuid.uuid4()), "tag": tag, "memory": None, "rationale": "parse error", "confidence": 0.0}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(LABELS_DIR.glob("*.json"))
    if not files:
        print(f"No label files found in {LABELS_DIR.resolve()}")
        return

    for f in files:
        labels = json.loads(f.read_text(encoding="utf-8"))
        tags = labels.get("tags", {})
        chunk_file = Path(labels["file"])
        if not chunk_file.exists():
            chunk_file = CHUNKS_DIR / (f.stem + ".txt")

        scene_text = chunk_file.read_text(encoding="utf-8-sig")

        for tag, active in tags.items():
            if not active:
                continue
            try:
                mem = ask_ollama_for_memory(MODEL, scene_text, tag)
            except Exception as e:
                print(f"[ERROR] {f.name} tag {tag}: {e}")
                continue

            payload = {
                "file": str(chunk_file),
                "model": MODEL,
                "scene": f.stem,
                **mem
            }

            # save one file per tag
            tag_dir = OUT_DIR / tag
            tag_dir.mkdir(parents=True, exist_ok=True)
            out_path = tag_dir / (f.stem + f"-{tag}.json")
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {out_path}")

        time.sleep(0.1)

if __name__ == "__main__":
    main()
