import sys, subprocess
from pathlib import Path
from typing import List

# ---------------- PATHS ----------------
ROOT          = Path(".").resolve()
SOURCE_FILE   = ROOT / "story_test_chapters"   # single source manuscript
CHUNKS_DIR    = ROOT / "text_chunks"
LABELS_DIR    = ROOT / "labels"
MEMORIES_ROOT = ROOT / "memories_by_chapter"
PROGRESS_FILE = ROOT / "progress.txt"

# ---------------- PROGRESS ----------------
def read_progress() -> int:
    if not PROGRESS_FILE.exists():
        PROGRESS_FILE.write_text("1", encoding="utf-8")
        print("No progress file, starting from 1")
        return 1
    try:
        return int(PROGRESS_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        PROGRESS_FILE.write_text("1", encoding="utf-8")
        return 1

def write_progress(val: int) -> None:
    PROGRESS_FILE.write_text(str(val), encoding="utf-8")
    print("advanced progress file")

# ---------------- HELPERS ----------------
def list_glob(dirpath: Path, patterns: List[str]) -> List[Path]:
    out = []
    for pat in patterns:
        out.extend(dirpath.glob(pat))
    # unique, sorted
    return sorted(set(out))

def chunks_for_chapter(ch: int) -> List[Path]:
    # Strict to your naming: {ch}-scene-{scene}-{part}.txt
    return list_glob(CHUNKS_DIR, [f"{ch}-scene-*.txt"])

def labels_for_chapter(ch: int) -> List[Path]:
    # Mirror chunk stems in labels dir
    return list_glob(LABELS_DIR, [f"{ch}-scene-*.json"])

def memories_present(ch: int) -> bool:
    target = MEMORIES_ROOT / f"{ch}-memories"
    if not target.exists():
        return False
    for sub in target.glob("*"):
        if sub.is_dir() and any(p.suffix.lower() == ".json" for p in sub.iterdir()):
            return True
    return False

# ---------------- STEPS ----------------
def ensure_split(chapter: int) -> List[Path]:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    existing = chunks_for_chapter(chapter)
    if existing:
        print(f"[split] found {len(existing)} chunk(s) for chapter {chapter}, skipping split.")
        return existing
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")

    print(f"[split] running splitter on {SOURCE_FILE.name}")
    # Try import first, fallback to subprocess. Your splitter should accept the source path.
    try:
        import splitter  # your splitter.py in the same folder
        if hasattr(splitter, "main"):
            splitter.main(str(SOURCE_FILE))
        elif hasattr(splitter, "split_file"):
            splitter.split_file(str(SOURCE_FILE))
        else:
            raise RuntimeError("splitter.py missing main()/split_file()")
    except Exception as e:
        print(f"[split] import failed ({e}); trying subprocess...")
        subprocess.run([sys.executable, "splitter.py", str(SOURCE_FILE)], check=True)

    existing = chunks_for_chapter(chapter)
    if not existing:
        raise RuntimeError(f"[split] splitter produced no chunks for chapter {chapter}")
    print(f"[split] produced {len(existing)} chunk(s).")
    return existing

def run_labeller():
    # Your labeller reads progress.txt, processes that chapter, then bumps progress by +1
    print("[label] running labeller_by_chapter.py")
    try:
        import labeller_by_chapter as lab
        if hasattr(lab, "main"):
            lab.main()
        else:
            raise RuntimeError("labeller_by_chapter missing main()")
    except Exception as e:
        print(f"[label] import failed ({e}); trying subprocess...")
        subprocess.run([sys.executable, "labeller_by_chapter.py"], check=True)

def run_extractor():
    # Your extractor reads progress and processes (progress-1), writing N-memories/
    print("[extract] running extract_memories_by_chapter.py")
    try:
        import extract_memories_by_chapter as ex
        if hasattr(ex, "main"):
            ex.main()
        else:
            raise RuntimeError("extract_memories_by_chapter missing main()")
    except Exception as e:
        print(f"[extract] import failed ({e}); trying subprocess...")
        subprocess.run([sys.executable, "extract_memories_by_chapter.py"], check=True)

# ---------------- ORCHESTRATOR ----------------
def main():
    chapter_ptr = read_progress()
    print(f"[progress] current pointer: chapter {chapter_ptr}")

    # 1) Ensure split exists for this chapter
    chunks = ensure_split(chapter_ptr)
    scenes = len(chunks)

    # 2) Label if missing
    existing_labels = labels_for_chapter(chapter_ptr)
    if len(existing_labels) >= scenes and scenes > 0:
        print(f"[label] labels already present for chapter {chapter_ptr}, skipping.")
    else:
        before = read_progress()
        if before != chapter_ptr:
            # if someone hand-edited progress, follow the file not your dreams
            chapter_ptr = before
            print(f"[label] aligning to progress.txt → chapter {chapter_ptr}")
        run_labeller()
        after = read_progress()
        if after != chapter_ptr + 1:
            print(f"[label] labeller did not advance progress; forcing to {chapter_ptr + 1}")
            write_progress(chapter_ptr + 1)

    # 3) Extract memories for just-labeled chapter (which is now progress-1)
    if memories_present(chapter_ptr):
        print(f"[extract] {chapter_ptr}-memories already exists and is populated, skipping.")
    else:
        run_extractor()
        if not memories_present(chapter_ptr):
            raise RuntimeError(f"[extract] finished, but {chapter_ptr}-memories has no JSON outputs")

    print(f"[done] Chapter {chapter_ptr}: split ✔ label ✔ extract ✔")
    print(f"[next] progress now {read_progress()} (next chapter to label)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
