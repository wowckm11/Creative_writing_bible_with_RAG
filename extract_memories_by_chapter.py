# extract_memories_by_chapter.py
import json, re, time, uuid
from pathlib import Path
from typing import Dict, Any, List, Callable
import ollama
import faiss
import numpy as np

# ---------------- CONFIG ----------------
MODEL = "qwen3:8b"
LABELS_DIR = Path("labels")
CHUNKS_DIR = Path("text_chunks")
MEMORIES_ROOT = Path("memories_by_chapter")
PROGRESS_FILE = Path("progress.txt")
SNAPSHOTS_DIR = Path("snapshots")             # RAG snapshots: ch-000/faiss.index + meta.json

MAX_PER_TAG = 8                    # cap per tag per scene
TOP_K_RAG = 5                      # how many prior snippets to retrieve
EMBED_MODEL = "nomic-embed-text"   # same as your indexer
MAX_CHARS_PER_QUERY = 2000
TEMPERATURE = 0.1

TAGS_WITH_DESC = {
    "important_memory_for_character": "foundational memories that might shape the character in the future",
    "relationship_change": "important changes to status quo between characters",
    "plot_fact": "an important happenstance that might be later referenced by someone",
    "magic_system": "concrete rules and explanations for how fantastical elements operate within the story",
    "fauna and flora": "names, species, physical descriptions and anything that is necessary to keep consistency across the story when referencing fauna and flora",
    "location_fact": "structural details about scenery that are necessary to keep consistency across the story when referencing past locations",
    "timeline": "mentions of time passing, in regards to the current in-world date, within the constraints of the scene",
    "foreshadowing": "long-term foreshadowing elements that should later be touched upon and utilised",
}

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

def files_for_chapter(chapter: int) -> List[Path]:
    cands = []
    cands += list(LABELS_DIR.glob(f"{chapter}-*.json"))
    cands += list(LABELS_DIR.glob(f"{chapter:03d}-*.json"))
    cands += list(LABELS_DIR.glob(f"ch-{chapter:03d}_*.json"))
    seen, out = set(), []
    for p in sorted(cands):
        if p not in seen:
            out.append(p); seen.add(p)
    return out

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

def _normalize_mem_item(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {"memory": None, "rationale": "bad item type", "confidence": 0.0}
    mem = obj.get("memory")
    rat = obj.get("rationale")
    try:
        conf = float(obj.get("confidence", 0))
    except Exception:
        conf = 0.0
    return {"memory": mem, "rationale": rat, "confidence": conf}

# ---------------- SNAPSHOT-AWARE RAG ----------------
def _embed_query(text: str) -> np.ndarray:
    t = text[:MAX_CHARS_PER_QUERY]
    v = ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"]
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def _build_ctx_retriever_for_chapter(current_chapter: int, top_k: int = TOP_K_RAG) -> Callable[[str], str] | None:
    """
    Returns a function(text)->context_block using snapshot ch-(current_chapter-1),
    or None if snapshot missing. Context is guidance-only, not a license to invent.
    """
    target = current_chapter - 1
    if target < 0:
        return None
    snap_dir = SNAPSHOTS_DIR / f"ch-{target:03d}"
    index_path = snap_dir / "faiss.index"
    meta_path = snap_dir / "meta.json"
    if not index_path.exists() or not meta_path.exists():
        print(f"[RAG] snapshot not found/complete for ch-{target:03d}")
        return None

    try:
        index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[RAG] failed loading snapshot ch-{target:03d}: {e}")
        return None

    def retrieve(text: str) -> str:
        if not text.strip():
            return ""
        q = _embed_query(text)
        D, I = index.search(q.reshape(1, -1), min(top_k, len(meta)))
        snippets = []
        for idx in I[0]:
            if 0 <= idx < len(meta):
                t = (meta[idx].get("text") or "").strip()
                if t:
                    snippets.append(t)
        if not snippets:
            return ""
        joined = "\n".join(f"- {s}" for s in snippets)
        return f"\n\nPrior context (chapters ≤ {target}; guidance only):\n{joined}\n"
    return retrieve

# ---------------- BATCH EXTRACTOR ----------------
def ask_ollama_for_memories_batch(
    model: str,
    scene_text: str,
    tag: str,
    count: int,
    ctx_block: str = "",
    retries: int = 2,
    temperature: float = TEMPERATURE,
) -> List[Dict[str, Any]]:
    """
    Ask model to extract up to `count` DISTINCT memories for the tag in one call.
    Context is optional; it should guide disambiguation but NOT introduce new facts.
    """
    count = max(0, min(int(count), MAX_PER_TAG))
    if count == 0:
        return []

    system = (
        "You extract DISTINCT, concrete memories for the given tag from the SCENE text. "
        "Use any prior context ONLY to disambiguate names/roles and avoid duplicates. "
        "Do NOT invent or import facts from context if the scene does not state them. "
        "Return ONLY JSON: {\"memories\": [{\"memory\": str, \"rationale\": str, \"confidence\": number}, ...]}."
    )
    prompt = (
        f"Tag: {tag}\n\n"
        f"This tag refers to: {TAGS_WITH_DESC.get(tag, 'n/a')}\n\n"
        f"Scene:\n{scene_text[:8000]}\n"
        f"{ctx_block or ''}"
        f"\nExtract up to {count} unique memories of this type.\n"
        "Rules:\n"
        "- Each memory 1–2 sentences, ≤45 words.\n"
        "- Include a short rationale citing scene details; confidence 0–1.\n"
        "- If a fact appears only in prior context but not in the scene, DO NOT include it.\n"
        "- No prose; JSON only as specified."
    )

    for attempt in range(retries + 1):
        res = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        content = _strip_code_fences(_remove_think_blocks(res["message"]["content"]))
        try:
            obj = json.loads(content)
            items = obj.get("memories", [])
            if not isinstance(items, list):
                items = []
            out = [_normalize_mem_item(it) for it in items]
            # Deduplicate by normalized memory text
            seen, uniq = set(), []
            for it in out:
                key = (it.get("memory") or "").strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                uniq.append(it)
            return uniq[:count]
        except Exception:
            if attempt == retries:
                raise ValueError(f"Model did not return valid JSON:\n{content}")
            time.sleep(0.5)
    return []

# ---------------- MAIN ----------------
def main():
    progress = read_progress()
    chapter_to_process = max(1, progress - 1)
    print(f"[progress] current={progress} → processing chapter {chapter_to_process}")

    label_files = files_for_chapter(chapter_to_process)
    if not label_files:
        print(f"No label files found for chapter {chapter_to_process}")
        return

    out_dir = MEMORIES_ROOT / f"{chapter_to_process}-memories"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build RAG retriever for ch-1
    ctx_fn = _build_ctx_retriever_for_chapter(chapter_to_process, top_k=TOP_K_RAG)

    for f in label_files:
        labels_doc = json.loads(f.read_text(encoding="utf-8"))
        tags_counts: Dict[str, Any] = labels_doc.get("tags", {})  # counts now
        chunk_file = Path(labels_doc["file"])
        if not chunk_file.exists():
            chunk_file = CHUNKS_DIR / (f.stem + ".txt")
        if not chunk_file.exists():
            print(f"[WARN] Missing chunk file for {f.name}")
            continue

        scene_text = chunk_file.read_text(encoding="utf-8-sig")
        # Get per-scene prior context block from snapshot
        ctx_block = ""
        if ctx_fn:
            try:
                ctx_block = ctx_fn(scene_text)
            except Exception as e:
                print(f"[RAG] retrieval failed: {e}")

        for tag, count in tags_counts.items():
            try:
                c = int(count)
            except Exception:
                c = 0
            if c <= 0:
                continue

            try:
                mems = ask_ollama_for_memories_batch(MODEL, scene_text, tag, c, ctx_block=ctx_block)
            except Exception as e:
                print(f"[ERROR] {f.name} tag {tag}: {e}")
                continue

            payload: Dict[str, Any] = {
                "file": str(chunk_file),
                "model": MODEL,
                "scene": f.stem,
                "chapter": chapter_to_process,
                "id": str(uuid.uuid4()),
                "tag": tag,
                "memories": mems,
            }

            tag_dir = out_dir / tag
            tag_dir.mkdir(parents=True, exist_ok=True)
            out_path = tag_dir / (f.stem + f"-{tag}.json")
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Wrote {out_path}")

        time.sleep(0.05)

if __name__ == "__main__":
    main()
