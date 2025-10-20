import json, re, time
from pathlib import Path
from typing import Dict, List, Any
import ollama

# ---------------- CONFIG ----------------
MODEL = "qwen3:8b"                 # Ollama model name
CHUNKS_DIR = Path("text_chunks")   # where your .txt chunks live
OUT_DIR = Path("labels")           # where to save JSON labels

# Tags with definitions (for prompt)
TAGS_WITH_DESC = [
    "important_memory_for_character: foundational memories that might shape the character in the future",
    "relationship_change: important changes to status quo between characters",
    "plot_fact: an important happenstance that might be later referenced by someone",
    "magic_system: concrete rules and explanations for how fantastical elements operate within the story",
    "fauna and flora: names, species, physical descriptions and anything that is neccesary to keep consistency across the story when referencing fauna and flora",
    "location_fact: structural details about scenery that are neccesary to keep consistency across the story when referencing past locations",
    "timeline: mentions of time passing, in regards to the current in-world date, within the constraints of the scene. Doesn't apply to references of events at other time, flashbacks and simmilar.",
    "foreshadowing: long-term foreshadowing elements that should later be touched upon and utilised",
]

# Plain tag keys (for JSON output)
TAGS = [
    "important_memory_for_character",
    "relationship_change",
    "plot_fact",
    "magic_system",
    "fauna and flora",
    "location_fact",
    "timeline",
    "foreshadowing",
]

TEMPERATURE = 0.0

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

def single_call(model: str, scene_text: str,
                tags_with_desc: List[str], tags_plain: List[str]) -> Dict[str, bool]:
    """Ask Ollama once for labels, return parsed dict."""
    system = (
        "You label fiction scenes for downstream RAG. "
        "You will be provided tags with descriptions. "
        "Return ONLY JSON: an object mapping each provided tag to true or false. "
        "Mark a tag true ONLY if the scene contains a concrete, extractable fact for that tag."
    )
    prompt = (
        "Tag definitions:\n" + "\n".join(f"- {t}" for t in tags_with_desc) +
        "\n\nScene:\n" + scene_text[:8000] +
        "\n\nOutput rules:\n"
        f"- Use exactly these keys: {tags_plain}\n"
        "- JSON only. No comments, no prose."
    )
    res = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        options={"temperature": TEMPERATURE},
    )
    content = _strip_code_fences(_remove_think_blocks(res["message"]["content"]))
    obj = json.loads(content)
    return {t: bool(obj.get(t, False)) for t in tags_plain}

def get_labels_with_tiebreaker(scene_text: str) -> (Dict[str, bool], List[Dict[str, bool]]):
    """Run twice, only do third run if first two disagree. Returns final labels and all raw runs."""
    r1 = single_call(MODEL, scene_text, TAGS_WITH_DESC, TAGS)
    time.sleep(0.05)
    r2 = single_call(MODEL, scene_text, TAGS_WITH_DESC, TAGS)

    if r1 == r2:
        return r1, [r1, r2]

    # tie-breaker
    r3 = single_call(MODEL, scene_text, TAGS_WITH_DESC, TAGS)
    final = {}
    for t in TAGS:
        votes_for_true = [r[t] for r in (r1, r2, r3)].count(True)
        votes_for_false = 3 - votes_for_true
        final[t] = True if votes_for_true > votes_for_false else False
    return final, [r1, r2, r3]

# ---------------- MAIN ----------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CHUNKS_DIR.glob("scene-*.txt"))
    if not files:
        print(f"No chunk files found in {CHUNKS_DIR.resolve()}")
        return

    for f in files:
        text = f.read_text(encoding="utf-8-sig").strip()
        if not text:
            continue

        print(f"Labelling {f.name}...")

        try:
            labels, raw_runs = get_labels_with_tiebreaker(text)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
            continue

        payload: Dict[str, Any] = {
            "file": str(f),
            "model": MODEL,
            "tags": labels,
            "raw_runs": raw_runs  # optional: keep all runs for inspection
        }

        out_path = OUT_DIR / (f.stem + ".json")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
