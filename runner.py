# runner.py
# Loops chapters: run one_chapter_crawler once per chapter, then rebuild RAG snapshot 0..N

import sys, json, time, re
from pathlib import Path
from typing import List, Dict, Any
import subprocess

import numpy as np
import faiss
import ollama

# ---------- PATHS ----------
ROOT               = Path(".").resolve()
SOURCE_FILE        = ROOT / "story_test_chapters"
CHUNKS_DIR         = ROOT / "text_chunks"
LABELS_DIR         = ROOT / "labels"
MEMORIES_ROOT      = ROOT / "memories_by_chapter"
BASELINE_DIR       = ROOT / "baseline_kb"         # optional curated base
SNAPSHOTS_DIR      = ROOT / "snapshots"
PROGRESS_FILE      = ROOT / "progress.txt"

# ---------- EMBEDDINGS ----------
EMBED_MODEL        = "nomic-embed-text"  # ollama pull nomic-embed-text
CONFIDENCE_MIN     = 0.50                # drop low-confidence memories from index
MAX_CHARS_PER_DOC  = 4000

# ---------- HELPERS ----------
def read_progress() -> int:
    if not PROGRESS_FILE.exists():
        PROGRESS_FILE.write_text("1", encoding="utf-8")
        return 1
    try:
        return int(PROGRESS_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        PROGRESS_FILE.write_text("1", encoding="utf-8")
        return 1

def chunks_for_chapter(ch: int) -> List[Path]:
    return sorted(CHUNKS_DIR.glob(f"{ch}-scene-*.txt"))

def labels_for_chapter(ch: int) -> List[Path]:
    return sorted(LABELS_DIR.glob(f"{ch}-scene-*.json"))

def memories_present(ch: int) -> bool:
    mdir = MEMORIES_ROOT / f"{ch}-memories"
    if not mdir.exists():
        return False
    for sub in mdir.glob("*"):
        if sub.is_dir() and any(p.suffix.lower() == ".json" for p in sub.iterdir()):
            return True
    return False

_CHAPTER_HEADER_RE = re.compile(r'^\s*(?:chapter|chap\.?)\s+(?:\d+|[ivxlcdm]+)\b', re.I | re.M)

def count_chapters_in_source() -> int:
    if not SOURCE_FILE.exists():
        return 0
    text = SOURCE_FILE.read_text(encoding="utf-8", errors="ignore")
    hits = _CHAPTER_HEADER_RE.findall(text)
    if hits:
        return len(hits)
    # fallback: infer from existing chunks
    nums = []
    for p in CHUNKS_DIR.glob("*-scene-*.txt"):
        try:
            n = int(p.name.split("-")[0])
            nums.append(n)
        except Exception:
            pass
    return max(nums) if nums else 0

def ensure_ollama_model(model: str) -> None:
    # nice-to-have; don’t explode if it fails
    try:
        ollama.list()
    except Exception:
        pass  # assume service is already up

def _embed(text: str) -> np.ndarray:
    t = text[:MAX_CHARS_PER_DOC]
    e = ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"]
    v = np.asarray(e, dtype="float32")
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v

# ---------- CORPUS GATHER ----------
def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _gather_baseline() -> List[Dict[str, Any]]:
    out = []
    if not BASELINE_DIR.exists():
        return out
    for p in BASELINE_DIR.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() == ".txt":
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            if txt:
                out.append({
                    "id": f"base:{p.as_posix()}",
                    "text": txt,
                    "chapter": 0,
                    "tag": "baseline",
                    "type": "world_rule",
                    "omniscient": True,
                    "pinned": True,
                    "source_path": p.as_posix()
                })
        elif p.suffix.lower() == ".json":
            obj = _read_json(p)
            txt = (obj.get("text") or obj.get("body") or obj.get("summary") or json.dumps(obj, ensure_ascii=False))[:MAX_CHARS_PER_DOC]
            if txt:
                out.append({
                    "id": obj.get("id") or f"base:{p.as_posix()}",
                    "text": txt,
                    "chapter": 0,
                    "tag": obj.get("tag","baseline"),
                    "type": obj.get("type","world_rule"),
                    "omniscient": bool(obj.get("omniscient", True)),
                    "pinned": bool(obj.get("pinned", True)),
                    "source_path": p.as_posix()
                })
    return out

def _gather_memories_to(chapter: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    root = MEMORIES_ROOT
    for n in range(0, chapter + 1):  # include 0-memories if present
        chap_dir = root / f"{n}-memories"
        if not chap_dir.exists():
            continue

        for p in chap_dir.rglob("*.json"):
            obj = _read_json(p)

            # Common fields
            base_id   = obj.get("id") or p.stem
            base_tag  = obj.get("tag", "unknown")
            base_type = obj.get("type", "memory")
            base_ch   = int(obj.get("chapter", n))
            base_omni = bool(obj.get("omniscient", False))
            base_pin  = bool(obj.get("pinned", False))

            # NEW FORMAT: array of sub-memories
            if isinstance(obj.get("memories"), list):
                for i, m in enumerate(obj["memories"], start=1):
                    mem_text = (m.get("memory") or "").strip()
                    if not mem_text:
                        continue
                    try:
                        conf = float(m.get("confidence", 0.0))
                    except Exception:
                        conf = 0.0
                    if n != 0 and conf < CONFIDENCE_MIN:
                        continue
                    out.append({
                        "id": f"{base_id}-{i}",
                        "text": mem_text[:MAX_CHARS_PER_DOC],
                        "chapter": base_ch,
                        "tag": base_tag,
                        "type": base_type,
                        "omniscient": base_omni,
                        "pinned": base_pin,
                        "source_path": p.as_posix(),
                    })
                continue  # done with this file

            # OLD FORMAT: single memory at top-level
            mem_text = (obj.get("memory") or "").strip()
            if not mem_text:
                continue
            try:
                conf = float(obj.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            if n != 0 and conf < CONFIDENCE_MIN:
                continue
            out.append({
                "id": base_id,
                "text": mem_text[:MAX_CHARS_PER_DOC],
                "chapter": base_ch,
                "tag": base_tag,
                "type": base_type,
                "omniscient": base_omni,
                "pinned": base_pin,
                "source_path": p.as_posix(),
            })
    return out

# ---------- INDEX REBUILD (CLEAN 0..N) ----------
def rebuild_snapshot(to_chapter: int) -> Path:
    ensure_ollama_model(EMBED_MODEL)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    snap_dir = SNAPSHOTS_DIR / f"ch-{to_chapter:03d}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    corpus = []
    corpus += _gather_baseline()
    corpus += _gather_memories_to(to_chapter)
    if not corpus:
        raise RuntimeError(f"[index] nothing to index up to chapter {to_chapter}")

    vecs = []
    for item in corpus:
        try:
            v = _embed(item["text"])
            vecs.append(v)
        except Exception as e:
            print(f"[index] embed failed for {item.get('id')} ({e}); skipping")
            continue

    X = np.vstack(vecs).astype("float32")
    index = faiss.IndexFlatIP(X.shape[1]); index.add(X)

    faiss.write_index(index, str(snap_dir / "faiss.index"))
    meta = corpus
    (snap_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "chapter": to_chapter,
        "entries": len(meta),
        "embedding_model": EMBED_MODEL,
        "confidence_min": CONFIDENCE_MIN,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "baseline_dir": BASELINE_DIR.as_posix(),
            "memories_root": MEMORIES_ROOT.as_posix()
        }
    }
    (snap_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[index] snapshot built → {snap_dir}")
    return snap_dir

# ---------- ONE-CHAPTER RUN ----------
def run_one_chapter_once() -> None:
    # Call your one_chapter_crawler in-process if importable, else subprocess.
    try:
        import one_chapter_crawler as occ
        if hasattr(occ, "main"):
            occ.main()
        else:
            raise RuntimeError("one_chapter_crawler missing main()")
    except Exception as e:
        print(f"[runner] import failed ({e}); trying subprocess...")
        subprocess.run([sys.executable, "one_chapter_crawler.py"], check=True)

# ---------- LOOP ----------
def main():
    total_in_source = count_chapters_in_source()
    if total_in_source == 0:
        print("[runner] Could not determine chapter count; will loop until crawler stops producing output.")

    while True:
        cur = read_progress()
        print(f"\n[runner] === pass for chapter {cur} ===")

        # If source has a known limit and we’re past it, stop.
        if total_in_source and cur > total_in_source:
            print(f"[runner] Reached end of source (>{total_in_source}). Stopping.")
            break

        # If we already have memories for this chapter and next chapter has no chunks, consider done.
        if memories_present(cur) and not chunks_for_chapter(cur):
            print(f"[runner] No chunks for chapter {cur}. Done.")
            break

        before = (len(labels_for_chapter(cur)), memories_present(cur))
        try:
            run_one_chapter_once()  # split → label → extract for chapter cur
        except subprocess.CalledProcessError as e:
            print(f"[runner] crawler error: {e}. Stopping.")
            break
        except Exception as e:
            print(f"[runner] crawler exception: {e}. Stopping.")
            break

        # After crawler, we expect memories for 'cur' and progress advanced to cur+1
        if not memories_present(cur):
            print(f"[runner] No memories found for chapter {cur} after crawler. Stopping.")
            break

        # Clean rebuild snapshot up to N=cur
        try:
            rebuild_snapshot(cur)
        except Exception as e:
            print(f"[runner] index rebuild failed for ch {cur}: {e}")
            # don’t die; you may want to continue to next chapters, but safest is to stop
            break

        # Termination condition: if we know total chapters and just built the last one, exit.
        nxt = read_progress()
        if total_in_source and nxt > total_in_source:
            print(f"[runner] Finished last chapter ({cur}).")
            break

        # If next chapter has no chunks and source is finite, we’ll stop on next loop naturally.
        print(f"[runner] Next pointer: chapter {nxt}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
