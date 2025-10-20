# label_chunks_tiebreak_rag_counts.py
import json, re, time
from pathlib import Path
from typing import Dict, List, Any
import ollama
import faiss
import numpy as np

# ---------------- CONFIG ----------------
MODEL = "qwen3:8b"
CHUNKS_DIR = Path("text_chunks")
OUT_DIR = Path("labels")
PROGRESS_FILE = Path("progress.txt")
SNAPSHOTS_DIR = Path("snapshots")

USE_BASELINE_RAG = True
TOP_K_BASELINE = 5
EMBED_MODEL = "nomic-embed-text"
MAX_CHARS_PER_QUERY = 2000
TEMPERATURE = 0.0

# ---------------- TAGS ----------------
TAGS_WITH_DESC = [
    "important_memory_for_character: foundational memories that might shape the character in the future",
    "relationship_change: important changes to status quo between characters",
    "plot_fact: an important happenstance that might be later referenced by someone",
    "magic_system: concrete rules and explanations for how fantastical elements operate within the story",
    "world_fact: concrete rules of the world, that matter disregarding whether characters know about them or not.",
    "fauna and flora: names, species, physical descriptions and anything that are necessary to keep consistency across the story when referencing fauna and flora",
    "location_fact: structural details about scenery that are necessary to keep consistency across the story when referencing past locations",
    "timeline: mentions of time passing, in regards to the current in-world date, within the constraints of the scene. Doesn't apply to references of events at other time, flashbacks, and similar.",
    "foreshadowing: long-term foreshadowing elements that should later be touched upon and utilised",
]
TAGS = [
    "important_memory_for_character",
    "relationship_change",
    "plot_fact",
    "magic_system",
    "world_fact",
    "fauna and flora",
    "location_fact",
    "timeline",
    "foreshadowing",
]

# ---------------- PROGRESS ----------------
def read_progress() -> int:
    if not PROGRESS_FILE.exists():
        PROGRESS_FILE.write_text("1", encoding="utf-8")
        return 1
    try:
        return int(PROGRESS_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        PROGRESS_FILE.write_text("1", encoding="utf-8")
        return 1

def write_progress(next_chapter: int) -> None:
    PROGRESS_FILE.write_text(str(next_chapter), encoding="utf-8")

def files_for_chapter(chapter: int) -> List[Path]:
    pats = [f"{chapter}-*.txt", f"{chapter:03d}-*.txt", f"ch-{chapter:03d}_*.txt"]
    out = []
    for p in pats:
        out.extend(CHUNKS_DIR.glob(p))
    return sorted(set(out))

# ---------------- RAG ----------------
def _embed_query(text: str) -> np.ndarray:
    t = text[:MAX_CHARS_PER_QUERY]
    v = ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"]
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _rag_context_for_chapter(current_chapter: int, top_k: int = 5):
    target = current_chapter - 1
    if target < 0:
        return None
    snap_dir = SNAPSHOTS_DIR / f"ch-{target:03d}"
    if not snap_dir.exists():
        print(f"[RAG] no snapshot for ch-{target}")
        return None
    index_path = snap_dir / "faiss.index"
    meta_path = snap_dir / "meta.json"
    if not index_path.exists() or not meta_path.exists():
        print(f"[RAG] incomplete snapshot for ch-{target}")
        return None
    try:
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[RAG] failed to load snapshot {target}: {e}")
        return None

    def retrieve(text: str) -> str:
        q = _embed_query(text)
        D, I = index.search(q.reshape(1, -1), min(top_k, len(meta)))
        retrieved = []
        for idx in I[0]:
            if 0 <= idx < len(meta):
                t = meta[idx].get("text") or ""
                if t.strip():
                    retrieved.append(t.strip())
        if not retrieved:
            return ""
        joined = "\n".join(f"- {r}" for r in retrieved)
        return f"\n\nPrior context (from chapters ≤{target}):\n{joined}\n"
    return retrieve

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

# ---------------- LABELLER ----------------
def single_call(model: str, scene_text: str, tags_with_desc: List[str], tags_plain: List[str], ctx_fn=None) -> Dict[str, int]:
    ctx_block = ""
    if ctx_fn:
        try:
            ctx_block = ctx_fn(scene_text)
        except Exception as e:
            print(f"[RAG] retrieval failed: {e}")
            ctx_block = ""

    system = (
        "You are labelling a scene for downstream RAG. "
        "Return ONLY JSON where each key is one of the provided tags, "
        "and the value is an integer count of how many *distinct* occurrences "
        "of that type appear in the scene. "
        "If the tag does not appear, set it to 0. "
        "Do not include anything else in the output."
    )
    tag_defs = "\n".join(f"- {t}" for t in tags_with_desc)
    prompt = (
        "Tag definitions:\n" + tag_defs +
        "\n\nScene:\n" + scene_text[:8000] +
        (ctx_block or "") +
        "\n\nOutput rules:\n"
        f"- Keys: {tags_plain}\n"
        "- Values: integer counts (0 if not present)\n"
        "- JSON only."
    )
    res = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        options={"temperature": TEMPERATURE},
    )
    content = _strip_code_fences(_remove_think_blocks(res["message"]["content"]))
    obj = json.loads(content)

    # enforce integers
    result = {}
    for t in tags_plain:
        val = obj.get(t, 0)
        try:
            result[t] = int(val)
        except Exception:
            result[t] = 0
    return result

def get_labels_with_tiebreaker(scene_text: str, ctx_fn=None):
    r1 = single_call(MODEL, scene_text, TAGS_WITH_DESC, TAGS, ctx_fn)
    time.sleep(0.05)
    r2 = single_call(MODEL, scene_text, TAGS_WITH_DESC, TAGS, ctx_fn)

    if r1 == r2:
        return r1, [r1, r2]

    r3 = single_call(MODEL, scene_text, TAGS_WITH_DESC, TAGS, ctx_fn)
    final = {}
    for t in TAGS:
        vals = [r1[t], r2[t], r3[t]]
        avg = round(sum(vals) / 3)
        final[t] = int(avg)
    return final, [r1, r2, r3]

# ---------------- MAIN ----------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chapter = read_progress()
    print(f"[progress] current chapter: {chapter}")

    files = files_for_chapter(chapter)
    if not files:
        print(f"[progress] no files for chapter {chapter}")
        return

    ctx_fn = _rag_context_for_chapter(chapter, top_k=TOP_K_BASELINE)

    for f in files:
        text = f.read_text(encoding="utf-8-sig").strip()
        if not text:
            continue

        print(f"Labelling {f.name}...")
        try:
            labels, raw_runs = get_labels_with_tiebreaker(text, ctx_fn)
        except Exception as e:
            print(f"[ERROR] {f.name}: {e}")
            continue

        payload = {
            "file": str(f),
            "model": MODEL,
            "tags": labels,   # now counts, not booleans
            "raw_runs": raw_runs,
        }
        out_path = OUT_DIR / (f.stem + ".json")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    write_progress(chapter + 1)
    print(f"[progress] advanced to chapter {chapter + 1}")

if __name__ == "__main__":
    main()
