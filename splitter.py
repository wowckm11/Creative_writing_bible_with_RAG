import re
from pathlib import Path
import sys

# Split on either "-=-=-" OR a chapter header, but ONLY chapter headers advance chapter_num
SEP_LINE_PATTERN = re.compile(r'^\s*-=-=-\s*$', flags=re.IGNORECASE)
CHAPTER_LINE_PATTERN = re.compile(r'^\s*(?:chapter|chap\.?)\s+(?:\d+|[ivxlcdm]+)\b.*$', flags=re.IGNORECASE)

OUT_DIR = Path("text_chunks")
WORD_LIMIT = 500

def split_long_chunk(text: str, limit: int) -> list[str]:
    """Split a long chunk into smaller chunks at newline boundaries, max ~limit words each."""
    words = text.split()
    if len(words) <= limit:
        return [text.strip()]
    lines = text.splitlines()
    result, current, current_count = [], [], 0
    for line in lines:
        lw = len(line.split())
        if current and current_count + lw > limit:
            result.append("\n".join(current).strip())
            current, current_count = [], 0
        current.append(line)
        current_count += lw
    if current:
        result.append("\n".join(current).strip())
    return [c for c in result if c]

def main(input_path: str) -> None:
    src = Path(input_path)
    if not src.exists():
        print(f"Input file not found: {src}")
        sys.exit(1)

    text = src.read_text(encoding="utf-8-sig")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chapter_num = 0
    scene_counter = 0
    written = 0
    buffer = []

    def flush_buffer():
        nonlocal scene_counter, written
        if not buffer or not "".join(buffer).strip():
            return
        scene_counter += 1
        scene_text = "\n".join(buffer).strip()
        buffer.clear()

        subchunks = split_long_chunk(scene_text, WORD_LIMIT)
        for j, sub in enumerate(subchunks, start=1):
            suffix = f"-{j:02d}" if len(subchunks) > 1 else ""
            out_file = OUT_DIR / f"{chapter_num}-scene-{scene_counter:03d}{suffix}.txt"
            # write the chapter header exactly as requested
            out_file.write_text(f"chapter {chapter_num}\n\n{sub.strip()}\n", encoding="utf-8")
            written += 1

    # Walk the file line-by-line so we know which delimiter we hit
    for line in text.splitlines():
        if CHAPTER_LINE_PATTERN.match(line):
            # end the current scene(s) under the current chapter
            flush_buffer()
            # now advance the chapter number ONLY for chapter headers
            chapter_num += 1
            continue
        if SEP_LINE_PATTERN.match(line):
            # plain scene break: flush, but DO NOT change chapter number
            flush_buffer()
            continue
        buffer.append(line)

    # flush whatever is left at EOF
    flush_buffer()

    print(f"Saved {written} chunk(s) to {OUT_DIR.resolve()} (final chapter number: {chapter_num})")

if __name__ == "__main__":
    main('story_test_chapters')
